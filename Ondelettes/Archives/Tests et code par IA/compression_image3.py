"""
wavelet_compression.py
======================

Compression d'image par ondelettes de Haar.

ARCHITECTURE
============
    WaveletBase   — classe abstraite : contrat de l'ondelette
    HaarWavelet   — implémentation Haar
    ImageCodec    — moteur de pipeline : transformée, seuillage, reconstruction,
                    affichage, sauvegarde

LIEN THÉORIE / PRATIQUE
========================
La Transformée en Ondelettes Continue (CWT) est une intégrale :

    W(a, b) = (1/√a) ∫ f(t) ψ*((t−b)/a) dt

Pour compresser, on utilise sa version DISCRÈTE (DWT), qui projette
l'image sur une base de fonctions :

    ψ_{j,k}(t) = 2^(j/2) ψ(2^j t − k)

calculée par le schéma de Mallat (1989) :
  1. Filtre passe-bas  h  → coefficients d'approximation (moyennes)
  2. Filtre passe-haut g  → coefficients de détail (différences)
  3. Sous-échantillonnage par 2 → moitié des données à chaque niveau
  → Complexité O(N), reconstruction parfaite.

Pour Haar : h = [1,1]/√2  g = [1,−1]/√2, ψ = indicatrice de [0,½)−[½,1).

COMPRESSION
===========
Après N niveaux de DWT :
  - Les coefficients de détail des zones LISSES sont proches de 0.
  - Le seuillage les met exactement à 0 → représentation CREUSE.
  - Taux de compression = (pixels originaux) / (coefficients non nuls × 6 o)
    (6 o = 2 o valeur int16 + 4 o position uint32, avant encodage entropique).
"""

from __future__ import annotations

import math
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _pad_pow2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Complète arr (2-D float) à la plus petite taille (2^p, 2^q) suffisante.
    Nécessaire car la DWT divise la taille par 2 à chaque niveau.
    """
    h, w = arr.shape
    H = 1 << math.ceil(math.log2(max(h, 2)))
    W = 1 << math.ceil(math.log2(max(w, 2)))
    out = np.zeros((H, W), dtype=float)
    out[:h, :w] = arr
    return out, (h, w)


def _unpad(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return arr[:shape[0], :shape[1]]


def _psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio en dB. > 40 dB : excellent. < 30 dB : visible."""
    mse = float(np.mean((orig.astype(float) - rec.astype(float)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _sparse_size_bytes(coeffs_channels: list[np.ndarray]) -> int:
    """
    Coût minimal de stockage sparse pour les 3 canaux :
      - chaque coeff non nul : 2 o (int16) + 4 o (position uint32) = 6 o
    C'est le plancher avant tout encodage entropique (Huffman, CABAC…).
    """
    nnz = sum(int(np.count_nonzero(c)) for c in coeffs_channels)
    return nnz * 6


def _png_size(img: np.ndarray) -> int:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return buf.tell()


# ─────────────────────────────────────────────────────────────────────────────
#  WaveletBase  —  classe abstraite
# ─────────────────────────────────────────────────────────────────────────────

class WaveletBase(ABC):
    """
    Contrat commun à toutes les ondelettes.

    Une sous-classe doit implémenter deux méthodes :
        _forward_1d(x)  →  coefficients  (même longueur que x)
        _inverse_1d(c)  →  signal reconstruit

    La DWT 2-D est construite par SÉPARABILITÉ (schéma de Mallat) :
        1. _forward_1d sur chaque LIGNE   → sous-bandes [L | H] en colonnes
        2. _forward_1d sur chaque COLONNE → sous-bandes 2-D [LL LH / HL HH]
    La reconstruction est l'opération symétrique en ordre inverse.

    À chaque niveau, on n'opère QUE sur le quadrant LL (coin ↖),
    ce qui produit la décomposition pyramidale multi-résolution.

    SUR super() :
        Si la sous-classe n'a pas de __init__ propre, elle hérite
        automatiquement celui de WaveletBase → pas besoin de super().
        Si elle ajoute des paramètres, elle écrit :
            def __init__(self, param, levels):
                super().__init__(levels)   ← délègue l'init commun
                self.param = param         ← puis ajoute le sien
    """

    name: str = "abstract"

    def __init__(self, levels: int = 4):
        self.levels = levels

    # ── Méthodes abstraites à implémenter ────────────────────────────────

    @abstractmethod
    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Transformée directe 1-D.
        x est un vecteur de longueur paire.
        Retourne un vecteur de même longueur :
            [:n//2] = approximation (passe-bas)
            [n//2:] = détail        (passe-haut)
        """

    @abstractmethod
    def _inverse_1d(self, c: np.ndarray) -> np.ndarray:
        """Transformée inverse 1-D. Reconstruction parfaite attendue."""

    # ── Transformée 2-D par séparabilité ─────────────────────────────────

    def _forward_2d_one_level(self, plane: np.ndarray) -> np.ndarray:
        """Un niveau de DWT 2-D sur la totalité du tableau plane."""
        h, w = plane.shape
        out = np.empty_like(plane)

        # Passe sur les lignes
        for i in range(h):
            out[i, :] = self._forward_1d(plane[i, :])

        # Passe sur les colonnes du résultat
        tmp = out.copy()
        for j in range(w):
            tmp[:, j] = self._forward_1d(out[:, j])

        return tmp

    def _inverse_2d_one_level(self, coeffs: np.ndarray) -> np.ndarray:
        """Un niveau de DWT 2-D inverse."""
        h, w = coeffs.shape
        tmp = np.empty_like(coeffs)

        # Inverse sur les colonnes d'abord
        for j in range(w):
            tmp[:, j] = self._inverse_1d(coeffs[:, j])

        out = np.empty_like(tmp)
        for i in range(h):
            out[i, :] = self._inverse_1d(tmp[i, :])

        return out

    # ── Décomposition multi-résolution ───────────────────────────────────

    def forward(self, plane: np.ndarray) -> np.ndarray:
        """
        Décomposition en self.levels niveaux (schéma pyramidal de Mallat).
        À chaque niveau, seul le quadrant LL (taille /2 en H et W) est
        re-décomposé.
        """
        c = plane.copy().astype(float)
        h, w = c.shape
        for _ in range(self.levels):
            c[:h, :w] = self._forward_2d_one_level(c[:h, :w])
            h //= 2
            w //= 2
        return c

    def inverse(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruction inverse en self.levels niveaux."""
        c = coeffs.copy()
        H, W = c.shape
        # On part du plus petit quadrant et on remonte
        h = H >> self.levels
        w = W >> self.levels
        for _ in range(self.levels):
            h *= 2
            w *= 2
            c[:h, :w] = self._inverse_2d_one_level(c[:h, :w])
        return c

    # ── Seuillage ─────────────────────────────────────────────────────────

    def threshold(self, coeffs: np.ndarray,
                  value: float, mode: str = "hard") -> np.ndarray:
        """
        Annule les coefficients de DÉTAIL en dessous de `value`.
        Le quadrant LL (approximation la plus grossière) est TOUJOURS préservé :
        il contient l'image basse résolution et ne doit pas être altéré.

        hard : c → 0                       si |c| < value  (coupure franche)
        soft : c → sign(c)·max(|c|−value,0)               (atténuation douce)

        Le seuillage soft réduit le bruit de Gibbs (artefacts de bord)
        mais peut rendre l'image légèrement floue.
        """
        out = coeffs.copy()
        H, W = out.shape
        lf_h = H >> self.levels     # hauteur du quadrant LL
        lf_w = W >> self.levels     # largeur  du quadrant LL

        # Masque : True = sous-bande de détail (à seuiller)
        mask = np.ones((H, W), dtype=bool)
        mask[:lf_h, :lf_w] = False

        if mode == "hard":
            out[mask & (np.abs(out) < value)] = 0.0

        elif mode == "soft":
            # Opération vectorielle, équivalente à :
            #   for each i,j in detail :
            #       s = sign(c); out[i,j] = s * max(|c|-value, 0)
            s = np.sign(out)
            magnitudes = np.abs(out)
            out[mask] = s[mask] * np.maximum(magnitudes[mask] - value, 0.0)

        else:
            raise ValueError(f"mode doit être 'hard' ou 'soft', pas '{mode}'")

        return out

    # ── Pipeline canal unique ─────────────────────────────────────────────

    def compress_plane(self, plane: np.ndarray,
                       thr: float, mode: str
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compresse un canal 2-D (uint8).
        Retourne (coefficients_seuillés, image_reconstruite_uint8).
        """
        arr, orig_shape = _pad_pow2(plane.astype(float))
        coeffs   = self.forward(arr)
        coeffs_t = self.threshold(coeffs, thr, mode)
        rec      = self.inverse(coeffs_t)
        rec      = np.clip(_unpad(rec, orig_shape), 0, 255).astype(np.uint8)
        return _unpad(coeffs_t, orig_shape), rec


# ─────────────────────────────────────────────────────────────────────────────
#  HaarWavelet
# ─────────────────────────────────────────────────────────────────────────────

class HaarWavelet(WaveletBase):
    """
    Ondelette de Haar (Alfred Haar, 1909).

    Ondelette mère :
        ψ(t) =  1  si t ∈ [0, ½)
               −1  si t ∈ [½, 1)
                0  sinon

    Filtres (orthonormaux) :
        h = [1,  1] / √2   ← passe-bas  (approximation = moyenne normalisée)
        g = [1, −1] / √2   ← passe-haut (détail = différence normalisée)

    Reconstruction parfaite car h et g sont orthogonaux :
        h·h + g·g = 1  et  h·g = 0

    La normalisation par √2 assure que l'énergie est conservée :
        ||x||² = ||approx||² + ||detail||²

    Lifting équivalent (2 opérations in-place) :
        d[k] = x[2k+1] − x[2k]              (prédiction)
        s[k] = x[2k] + d[k]/2               (mise à jour)
    Ce code utilise la forme filtre, plus lisible.
    """

    name = "Haar"

    # Pas de __init__ propre : on hérite directement WaveletBase.__init__(levels)

    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        assert n % 2 == 0, f"longueur impaire : {n}"

        s = 1.0 / math.sqrt(2)

        # Échantillons pairs et impairs
        evens = x[0::2]   # x[0], x[2], x[4], ...
        odds  = x[1::2]   # x[1], x[3], x[5], ...

        approx = (evens + odds) * s   # projection sur φ_{j,k} (moyenne)
        detail = (evens - odds) * s   # projection sur ψ_{j,k} (différence)

        out = np.empty(n, dtype=float)
        out[:n // 2] = approx
        out[n // 2:] = detail
        return out

    def _inverse_1d(self, c: np.ndarray) -> np.ndarray:
        n    = len(c)
        half = n // 2
        s    = 1.0 / math.sqrt(2)

        approx = c[:half]
        detail = c[half:]

        # Reconstruction des paires : formule inverse exacte
        out = np.empty(n, dtype=float)
        out[0::2] = (approx + detail) * s
        out[1::2] = (approx - detail) * s
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Rapport de compression
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionReport:
    """
    Métriques calculées après une compression.

    sparsity      : fraction de coefficients annulés par le seuillage.
                    Un codec entropique (Huffman, CABAC) exploite exactement
                    cette propriété : une représentation creuse se compresse bien.

    original_bytes: taille brute RGB (H×W×3 octets, sans aucune compression).
    sparse_bytes  : coût théorique de stockage sparse des coefficients
                    (6 octets par coeff non nul : 2 valeur + 4 position).
    png_bytes     : taille de l'original en PNG (compression sans perte, référence).
    ratio_sparse  : original_bytes / sparse_bytes  (gain sparse brut).
    ratio_png     : png_bytes / sparse_bytes       (gain réel vs PNG).

    PSNR          : qualité. > 40 dB = imperceptible, 35-40 dB = bon,
                    30-35 dB = légère dégradation, < 30 dB = visible.
    """
    wavelet_name   : str
    levels         : int
    threshold      : float
    threshold_mode : str
    psnr           : float
    sparsity       : float
    original_bytes : int
    sparse_bytes   : int
    png_bytes      : int
    ratio_sparse   : float = field(init=False)
    ratio_png      : float = field(init=False)

    def __post_init__(self):
        self.ratio_sparse = self.original_bytes / max(self.sparse_bytes, 1)
        self.ratio_png    = self.png_bytes       / max(self.sparse_bytes, 1)

    def __str__(self) -> str:
        sep = "─" * 46
        return "\n".join([
            "┌" + sep + "┐",
            f"│  Ondelette    : {self.wavelet_name:<28} │",
            f"│  Niveaux      : {self.levels:<28} │",
            f"│  Seuil        : {self.threshold:<28.2f} │",
            f"│  Mode         : {self.threshold_mode:<28} │",
            "├" + sep + "┤",
            f"│  PSNR         : {self.psnr:<26.2f} dB │",
            f"│  Sparsité     : {100*self.sparsity:<26.1f}  % │",
            "├" + sep + "┤",
            f"│  Brut RGB     : {self.original_bytes:<24} o   │",
            f"│  PNG original : {self.png_bytes:<24} o   │",
            f"│  Sparse (6o)  : {self.sparse_bytes:<24} o   │",
            f"│  Ratio brut   : {self.ratio_sparse:<26.2f} × │",
            f"│  Ratio vs PNG : {self.ratio_png:<26.2f} × │",
            "└" + sep + "┘",
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  ImageCodec  —  moteur de pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ImageCodec:
    """
    Moteur de compression par ondelettes.

    Gère la pipeline complète :
        chargement → DWT → seuillage → DWT inverse → métriques
        → affichage → sauvegarde

    Paramètres
    ----------
    source         : chemin image ou tableau numpy RGB uint8
    wavelet        : instance de WaveletBase (HaarWavelet ou autre)
    threshold      : seuil de coupure des coefficients de détail
    threshold_mode : 'hard' (coupure franche) ou 'soft' (atténuation)

    Exemple
    -------
        codec = ImageCodec("lenna.jpg", HaarWavelet(levels=5), threshold=20)
        print(codec.run())
        codec.show()
        codec.save("lenna_haar.png")
        codec.save("lenna_haar.jpg", jpeg_quality=90)
    """

    def __init__(self, source: str | Path | np.ndarray,
                 wavelet: WaveletBase,
                 threshold: float = 15.0,
                 threshold_mode: str = "hard"):
        self.wavelet        = wavelet
        self.threshold_val  = threshold
        self.threshold_mode = threshold_mode

        if isinstance(source, (str, Path)):
            self.original    = np.array(Image.open(source).convert("RGB"), dtype=np.uint8)
            self.source_name = Path(source).name
        else:
            self.original    = source.astype(np.uint8)
            self.source_name = "array"

        self.reconstructed      : Optional[np.ndarray]       = None
        self.coeffs_per_channel : list[np.ndarray]           = []
        self.report             : Optional[CompressionReport] = None

    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> CompressionReport:
        """Exécute la pipeline complète et retourne les métriques."""
        recs, coeffs = [], []
        for ch in range(3):
            coeff, rec = self.wavelet.compress_plane(
                self.original[:, :, ch],
                self.threshold_val,
                self.threshold_mode,
            )
            coeffs.append(coeff)
            recs.append(rec)

        self.reconstructed      = np.stack(recs, axis=2)
        self.coeffs_per_channel = coeffs

        all_c    = np.concatenate([c.ravel() for c in coeffs])
        sparsity = float(np.sum(all_c == 0) / all_c.size)
        H, W     = self.original.shape[:2]

        self.report = CompressionReport(
            wavelet_name   = self.wavelet.name,
            levels         = self.wavelet.levels,
            threshold      = self.threshold_val,
            threshold_mode = self.threshold_mode,
            psnr           = _psnr(self.original, self.reconstructed),
            sparsity       = sparsity,
            original_bytes = H * W * 3,
            sparse_bytes   = _sparse_size_bytes(coeffs),
            png_bytes      = _png_size(self.original),
        )
        return self.report

    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path, jpeg_quality: int = 92) -> Path:
        """
        Sauvegarde l'image reconstruite.
        Format déduit de l'extension : .jpg / .jpeg → JPEG, sinon → PNG.
        """
        if self.reconstructed is None:
            raise RuntimeError("Appeler .run() avant .save()")
        path = Path(path)
        img  = Image.fromarray(self.reconstructed)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            img.save(path, format="JPEG", quality=jpeg_quality)
        else:
            img.save(path, format="PNG")
        size = path.stat().st_size
        print(f"  Sauvegardé : {path}  ({size} octets, "
              f"{self.original.size / size:.2f}× vs brut RGB)")
        return path

    # ──────────────────────────────────────────────────────────────────────

    def show(self, save_fig: Optional[str | Path] = None) -> None:
        """
        Figure en 5 panneaux :
          ┌──────────┬──────────┬──────────┐
          │ Original │ Reconst. │  Erreur  │
          ├──────────┴──────────┼──────────┤
          │  Coefficients DWT   │  Histog. │
          └─────────────────────┴──────────┘

        Le panneau "Coefficients" montre log(1+|c|) en faux-couleurs :
          - coin ↖ (petit) = sous-bande LL = approximation basse résolution
          - reste           = sous-bandes de détail (LH, HL, HH)
          Zones sombres = coefficients proches de 0 = bien compressés.
        """
        if self.reconstructed is None:
            raise RuntimeError("Appeler .run() avant .show()")

        fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.36, wspace=0.14,
                                left=0.04, right=0.96,
                                top=0.89,  bottom=0.06)
        tkw = dict(color="#e8e8e8", fontsize=10, pad=6, fontfamily="monospace")

        # ── Original ──────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(self.original)
        ax0.set_title("Original", **tkw)
        ax0.axis("off")

        # ── Reconstruit ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.reconstructed)
        ax1.set_title(f"Reconstruit  (PSNR = {self.report.psnr:.1f} dB)", **tkw)
        ax1.axis("off")

        # ── Erreur absolue ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        err = np.abs(self.original.astype(int) -
                     self.reconstructed.astype(int)).mean(axis=2)
        im2 = ax2.imshow(err, cmap="inferno", vmin=0, vmax=max(float(err.max()), 1.0))
        ax2.set_title("Erreur absolue (moy. RGB)", **tkw)
        ax2.axis("off")
        cb = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="intensité")
        cb.ax.yaxis.label.set(color="#888888", fontsize=8)
        cb.ax.tick_params(colors="#888888", labelsize=7)

        # ── Coefficients (luminance) ──────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0:2])
        luma = (0.299 * self.coeffs_per_channel[0] +
                0.587 * self.coeffs_per_channel[1] +
                0.114 * self.coeffs_per_channel[2])
        ax3.imshow(np.log1p(np.abs(luma)), cmap="viridis", interpolation="nearest")
        ax3.set_title(
            f"Coefficients DWT – {self.wavelet.name}  "
            f"({self.wavelet.levels} niveaux)  "
            f"sparsité = {100*self.report.sparsity:.1f}%   "
            f"ratio brut = {self.report.ratio_sparse:.2f}×   "
            f"ratio PNG = {self.report.ratio_png:.2f}×",
            **tkw)
        ax3.axis("off")

        # Annotation des sous-bandes LL / LH / HL / HH
        H, W = luma.shape
        lh = H >> self.wavelet.levels
        lw = W >> self.wavelet.levels
        for label, x0, y0, dx, dy in [
            ("LL", 0,  0,  lw,   lh),
            ("LH", lw, 0,  W-lw, lh),
            ("HL", 0,  lh, lw,   H-lh),
            ("HH", lw, lh, W-lw, H-lh),
        ]:
            ax3.text(x0 + dx // 2, y0 + dy // 2, label,
                     color="white", fontsize=8, ha="center", va="center",
                     alpha=0.6, fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.2",
                               fc="black", alpha=0.35, ec="none"))

        # ── Histogramme des coefficients de détail ────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        # On n'affiche que les détails (hors quadrant LL)
        detail_mask = np.ones(luma.shape, dtype=bool)
        detail_mask[:lh, :lw] = False
        detail_vals = luma[detail_mask].ravel()
        nz = detail_vals[detail_vals != 0]
        ax4.hist(nz, bins=200, color="#4fc3f7", alpha=0.85, log=True)
        ax4.axvline( self.threshold_val, color="#ff5252", lw=1.5,
                     label=f"±seuil = {self.threshold_val}")
        ax4.axvline(-self.threshold_val, color="#ff5252", lw=1.5)
        ax4.set_facecolor("#161616")
        ax4.tick_params(colors="#888888", labelsize=7)
        for sp in ax4.spines.values():
            sp.set_color("#333333")
        ax4.set_title("Détails DWT (log)", **tkw)
        ax4.set_xlabel("valeur", color="#888888", fontsize=8)
        ax4.set_ylabel("occurrences", color="#888888", fontsize=8)
        ax4.legend(fontsize=7, labelcolor="#cccccc",
                   facecolor="#1e1e1e", edgecolor="#333333")

        # ── Titre global ──────────────────────────────────────────────────
        fig.suptitle(
            f"{self.source_name}  ·  {self.wavelet.name}  ·  "
            f"seuil {self.threshold_mode} = {self.threshold_val}  ·  "
            f"PSNR = {self.report.psnr:.1f} dB  ·  "
            f"ratio PNG→sparse = {self.report.ratio_png:.2f}×",
            color="#e8e8e8", fontsize=11, fontfamily="monospace", y=0.96)

        plt.tight_layout()
        if save_fig:
            fig.savefig(save_fig, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Comparaison Haar à différents seuils
# ─────────────────────────────────────────────────────────────────────────────

def sweep_threshold(source: str | Path,
                    thresholds: list[float],
                    levels: int = 5,
                    save_fig: Optional[str | Path] = None) -> None:
    """
    Courbes PSNR, ratio sparse et sparsité en fonction du seuil.
    Permet de choisir le bon compromis qualité / compression.
    """
    original = np.array(Image.open(source).convert("RGB"), dtype=np.uint8)
    data: list[tuple[float, float, float, float]] = []

    print(f"  Balayage {len(thresholds)} seuils …", end="", flush=True)
    for thr in thresholds:
        rep = ImageCodec(original.copy(), HaarWavelet(levels=levels),
                         threshold=thr).run()
        data.append((thr, rep.psnr, rep.ratio_png, 100 * rep.sparsity))
    print(" OK")

    thrs      = [d[0] for d in data]
    psnrs     = [d[1] for d in data]
    ratios    = [d[2] for d in data]
    sparsity  = [d[3] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    color = "#4fc3f7"

    for ax, ys, ylabel, title in [
        (axes[0], psnrs,    "PSNR (dB)",           "Qualité"),
        (axes[1], ratios,   "Ratio PNG→sparse (×)", "Taux de compression"),
        (axes[2], sparsity, "Sparsité (%)",          "Coefficients nuls"),
    ]:
        ax.plot(thrs, ys, "-o", color=color, ms=4, lw=2)
        ax.set_facecolor("#161616")
        ax.set_xlabel("Seuil", color="#888888", fontsize=9)
        ax.set_ylabel(ylabel, color="#888888", fontsize=9)
        ax.set_title(title, color="#e8e8e8", fontsize=10, fontfamily="monospace")
        ax.tick_params(colors="#888888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.grid(alpha=0.15, color="#444444")

    # Repères qualité
    for y, lbl, c in [(40, "40 dB excellent", "#69f0ae"),
                       (35, "35 dB imperceptible", "#ffcc02"),
                       (30, "30 dB visible", "#ff5252")]:
        axes[0].axhline(y, color=c, lw=0.8, ls="--", alpha=0.7, label=lbl)
    axes[0].legend(fontsize=7, labelcolor="#cccccc",
                   facecolor="#1e1e1e", edgecolor="#333333")

    fig.suptitle(
        f"Haar  ·  {levels} niveaux  ·  {Path(source).name}",
        color="#e8e8e8", fontsize=12, fontfamily="monospace")
    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
img = "moon.jpeg"
if __name__ == "__main__":
    import sys
    import time

    IMAGE = sys.argv[1] if len(sys.argv) > 1 else img

    print("=" * 52)
    print("  Compression Haar")
    print("=" * 52)
    t0    = time.perf_counter()
    codec = ImageCodec(IMAGE, HaarWavelet(levels=5), threshold=50)
    rep   = codec.run()
    print(f"  Durée : {time.perf_counter() - t0:.2f} s")
    print(rep)
    codec.show()
    codec.save("lenna_haar.png")
    codec.save("lenna_haar.jpg", jpeg_quality=90)

    print("\n[Balayage des seuils]")
    sweep_threshold(IMAGE, thresholds=[5, 10, 20, 35, 50, 80, 120], levels=5)