"""
wavelet_compression.py
======================

Module de compression d'image par ondelettes.

Deux familles d'ondelettes sont implémentées :

1. **Haar** (ondelette la plus simple)
   - Ondelette mère : ψ(t) = 1 sur [0, 0.5), -1 sur [0.5, 1), 0 sinon
   - Fonction d'échelle : φ(t) = 1 sur [0, 1)
   - La décomposition discrète revient exactement à ce qu'on faisait
     à la main : moyennes et différences sur des blocs 2×2.

2. **Daubechies D4** (base de JPEG 2000, ondelette à support compact)
   - Coefficients de filtre obtenus par la théorie des moments nuls.
   - ψ possède N=2 moments nuls → comprime mieux les signaux réguliers.
   - On applique le filtre en lignes puis en colonnes (transformée séparable).
   - C'est la "vraie" voie mathématique : on projette l'image sur une base
     orthonormale de L²(R²) construite par dilatations/translations de ψ.

Architecture
------------
    WaveletBase          – interface commune (compress / decompress / threshold)
    ├── HaarWavelet      – implémentation Haar 2-D lifté en place
    └── Daubechies4      – implémentation DB4 via banc de filtres séparable
    ImageCodec           – pipeline complet : charge, encode, seuille, décode, affiche
    CompressionReport    – métriques (PSNR, SSIM-simplifié, taux de compression)
"""

from __future__ import annotations
import csv
import math
import struct
import zlib
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires bas niveau
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_power2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Complète arr (2-D) à la taille (2^p × 2^q) la plus petite suffisante."""
    h, w = arr.shape
    H = 1 << math.ceil(math.log2(h)) if h > 1 else 1
    W = 1 << math.ceil(math.log2(w)) if w > 1 else 1
    padded = np.zeros((H, W), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded, (h, w)


def _unpad(arr: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    return arr[:h, :w]


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio en dB (max_val = 255)."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0 ** 2 / mse)


def _compressed_size_bytes(arr: np.ndarray) -> int:
    """Taille après compression zlib (proxy du codec entropique réel)."""
    raw = arr.astype(np.int16).tobytes()
    return len(zlib.compress(raw, level=9))


# ─────────────────────────────────────────────────────────────────────────────
#  Classe de base
# ─────────────────────────────────────────────────────────────────────────────

class WaveletBase(ABC):
    """
    Interface abstraite pour une ondelette 2-D appliquée canal par canal.

    Un sous-classement doit implémenter :
        _forward_1d(signal)  → coefficients
        _inverse_1d(coeffs)  → signal reconstruit

    La transformée 2-D est obtenue par séparabilité :
        W2D = W_lignes ∘ W_colonnes   (standard multi-résolution)
    """

    name: str = "abstract"

    # ------------------------------------------------------------------
    # Transformée 1-D (à surcharger)
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward_1d(self, signal: np.ndarray) -> np.ndarray:
        """Transformée directe sur un vecteur 1-D (longueur puissance de 2)."""

    @abstractmethod
    def _inverse_1d(self, coeffs: np.ndarray) -> np.ndarray:
        """Transformée inverse sur un vecteur 1-D."""

    # ------------------------------------------------------------------
    # Transformée 2-D par séparabilité (n niveaux)
    # ------------------------------------------------------------------

    def _forward_2d_level(self, plane: np.ndarray) -> np.ndarray:
        """Un seul niveau de décomposition 2-D sur la totalité du tableau."""
        # Lignes
        out = np.apply_along_axis(self._forward_1d, axis=1, arr=plane)
        # Colonnes
        out = np.apply_along_axis(self._forward_1d, axis=0, arr=out)
        return out

    def _inverse_2d_level(self, coeffs: np.ndarray) -> np.ndarray:
        out = np.apply_along_axis(self._inverse_1d, axis=0, arr=coeffs)
        out = np.apply_along_axis(self._inverse_1d, axis=1, arr=out)
        return out

    def forward(self, plane: np.ndarray, levels: int) -> np.ndarray:
        """
        Décomposition multi-résolution sur `levels` niveaux.

        À chaque niveau n, on ne transforme que le quadrant basse-fréquence
        (coins supérieur-gauche de taille H/2^n × W/2^n), exactement
        comme dans le schéma pyramidal de Mallat (1989).
        """
        coeffs = plane.copy().astype(float)
        h, w = coeffs.shape
        for _ in range(levels):
            sub = coeffs[:h, :w]
            sub_t = self._forward_2d_level(sub)
            coeffs[:h, :w] = sub_t
            h //= 2
            w //= 2
        return coeffs

    def inverse(self, coeffs: np.ndarray, levels: int) -> np.ndarray:
        """Reconstruction multi-résolution inverse (schéma de Mallat)."""
        rec = coeffs.copy()
        H, W = rec.shape
        h = H >> levels
        w = W >> levels
        for _ in range(levels):
            h *= 2
            w *= 2
            sub = rec[:h, :w]
            rec[:h, :w] = self._inverse_2d_level(sub)
        return rec

    # ------------------------------------------------------------------
    # Seuillage (hard ou soft)
    # ------------------------------------------------------------------

    def threshold(
        self,
        coeffs: np.ndarray,
        value: float,
        levels: int,
        mode: str = "hard",
    ) -> np.ndarray:
        """
        Seuillage des sous-bandes hautes fréquences uniquement.

        On préserve le quadrant basse-fréquence (approximation au niveau
        le plus grossier) et on annule les détails en dessous de `value`.

        mode = 'hard' : coeff → 0 si |coeff| < value
        mode = 'soft' : coeff → sign(c) * max(|c| - value, 0)
        """
        out = coeffs.copy()
        h, w = out.shape
        lf_h = h >> levels
        lf_w = w >> levels

        mask = np.ones((h, w), dtype=bool)
        mask[:lf_h, :lf_w] = False          # basse fréquence = inchangée

        if mode == "hard":
            out[mask & (np.abs(out) < value)] = 0.0
        elif mode == "soft":
            sign = np.sign(out)
            out[mask] = sign[mask] * np.maximum(np.abs(out[mask]) - value, 0.0)
        else:
            raise ValueError(f"mode doit être 'hard' ou 'soft', pas '{mode}'")
        return out

    # ------------------------------------------------------------------
    # Pipeline complet sur un canal niveaux de gris
    # ------------------------------------------------------------------

    def compress_plane(
        self,
        plane: np.ndarray,
        levels: int,
        threshold_value: float,
        threshold_mode: str = "hard",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (coefficients_seuillés, image_reconstruite).
        `plane` est un tableau 2-D uint8.
        """
        arr, original_shape = _pad_to_power2(plane.astype(float))
        coeffs = self.forward(arr, levels)
        coeffs_t = self.threshold(coeffs, threshold_value, levels, threshold_mode)
        rec = self.inverse(coeffs_t, levels)
        rec = _unpad(rec, original_shape)
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        return _unpad(coeffs_t, original_shape), rec


# ─────────────────────────────────────────────────────────────────────────────
#  Ondelette de Haar
# ─────────────────────────────────────────────────────────────────────────────

class HaarWavelet(WaveletBase):
    """
    Ondelette de Haar – la plus ancienne (Alfred Haar, 1909).

    Ondelette mère :
        ψ(t) =  1   si t ∈ [0, 1/2)
               -1   si t ∈ [1/2, 1)
                0   sinon

    Filtres de décomposition :
        h₀ = [1/√2,  1/√2]   (passe-bas, approximation)
        h₁ = [1/√2, -1/√2]   (passe-haut, détail)

    La transformée discrète revient à calculer des moyennes et différences
    sur des paires de valeurs consécutives — d'où sa simplicité.
    """

    name = "Haar"

    def _forward_1d(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        assert n & (n - 1) == 0 and n >= 2, "La longueur doit être une puissance de 2"
        out = np.empty_like(signal)
        half = n // 2
        # Approximation (basse fréquence) : moyenne normalisée
        out[:half] = (signal[0::2] + signal[1::2]) / math.sqrt(2)
        # Détail (haute fréquence) : différence normalisée
        out[half:] = (signal[0::2] - signal[1::2]) / math.sqrt(2)
        return out

    def _inverse_1d(self, coeffs: np.ndarray) -> np.ndarray:
        n = len(coeffs)
        half = n // 2
        approx = coeffs[:half]
        detail = coeffs[half:]
        out = np.empty_like(coeffs)
        out[0::2] = (approx + detail) / math.sqrt(2)
        out[1::2] = (approx - detail) / math.sqrt(2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Ondelette de Daubechies D4
# ─────────────────────────────────────────────────────────────────────────────

class Daubechies4(WaveletBase):
    """
    Ondelette de Daubechies à 4 coefficients (D4 / db2).

    Ingrid Daubechies (1988) a construit la première famille d'ondelettes
    orthogonales à support compact avec des moments nuls.  D4 possède
    N = 2 moments nuls :  ∫ t^k ψ(t) dt = 0  pour k = 0, 1

    Cela signifie que ψ est orthogonale aux polynômes de degré ≤ 1 :
    les régions lisses d'une image sont très bien compressées (peu de
    coefficients significatifs).

    Coefficients de filtre passe-bas h₀ (orthonormaux, calculés via la
    condition de régularité de Daubechies) :
        h₀ = [(1+√3)/(4√2),  (3+√3)/(4√2),  (3-√3)/(4√2),  (1-√3)/(4√2)]

    Le filtre passe-haut h₁ est obtenu par modulation en quadrature :
        h₁[k] = (-1)^k h₀[3-k]

    La transformée est périodisée aux bords (convention JPEG 2000).
    """

    name = "Daubechies D4"

    _S3 = math.sqrt(3)
    _S2 = math.sqrt(2)

    # Coefficients de filtre passe-bas (analyse)
    H0: np.ndarray = np.array([
        (1 + _S3) / (4 * _S2),
        (3 + _S3) / (4 * _S2),
        (3 - _S3) / (4 * _S2),
        (1 - _S3) / (4 * _S2),
    ])

    # Filtre passe-haut par alternance de signes (QMF)
    H1: np.ndarray = np.array([
         (1 - _S3) / (4 * _S2),
        -(3 - _S3) / (4 * _S2),
         (3 + _S3) / (4 * _S2),
        -(1 + _S3) / (4 * _S2),
    ])

    # Filtres de synthèse (reconstruction) : h̃₀[k] = h₀[-k], h̃₁[k] = h₁[-k]
    G0: np.ndarray = H0[::-1].copy()
    G1: np.ndarray = H1[::-1].copy()

    def _forward_1d(self, signal: np.ndarray) -> np.ndarray:
        """
        Banc de filtres d'analyse avec sous-échantillonnage par 2.

        Pour un signal x de longueur N :
            approx[k] = Σ_n h₀[n-2k] x[n]   (passe-bas décimé)
            detail[k] = Σ_n h₁[n-2k] x[n]   (passe-haut décimé)

        Extension périodique pour gérer les bords.
        """
        n = len(signal)
        # Convolution périodique et sous-échantillonnage
        approx = np.array([
            sum(self.H0[m] * signal[(2 * k + m) % n] for m in range(4))
            for k in range(n // 2)
        ])
        detail = np.array([
            sum(self.H1[m] * signal[(2 * k + m) % n] for m in range(4))
            for k in range(n // 2)
        ])
        return np.concatenate([approx, detail])

    def _inverse_1d(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Banc de filtres de synthèse : sur-échantillonnage + filtrage.

            x[n] = Σ_k g₀[n-2k] approx[k] + Σ_k g₁[n-2k] detail[k]
        """
        n = len(coeffs)
        half = n // 2
        approx = coeffs[:half]
        detail = coeffs[half:]
        out = np.zeros(n)
        for k in range(half):
            for m in range(4):
                idx = (2 * k + m) % n
                out[idx] += self.G0[m] * approx[k] + self.G1[m] * detail[k]
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Ondelette de Morlet (analyse temps-fréquence, non orthogonale)
# ─────────────────────────────────────────────────────────────────────────────

class MorletWavelet:
    """
    Ondelette de Morlet – utilisée pour l'analyse temps-fréquence, pas
    directement pour la compression (elle n'est pas orthogonale ni à support
    compact).  On l'inclut ici pour visualiser le spectre en ondelettes
    d'un signal 1-D extrait d'une ligne de l'image.

    Définition continue :
        ψ(t) = π^(-1/4) exp(iω₀t) exp(-t²/2)
    avec ω₀ = 6 (standard).

    La Transformée en Ondelettes Continue (CWT) est :
        W(a, b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt

    On discrétise sur une grille logarithmique en échelle `a`.
    """

    def __init__(self, omega0: float = 6.0):
        self.omega0 = omega0

    def wavelet(self, t: np.ndarray) -> np.ndarray:
        """ψ(t) complexe."""
        c = math.pi ** (-0.25)
        return c * np.exp(1j * self.omega0 * t) * np.exp(-0.5 * t ** 2)

    def cwt(
        self,
        signal: np.ndarray,
        scales: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Transformée en ondelettes continue par convolution dans le domaine
        fréquentiel (algorithme O(N log N)).

        Retourne un tableau complexe de forme (len(scales), len(signal)).
        """
        n = len(signal)
        sig_fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, dt)   # fréquences angulaires / (2π)
        omega = 2 * math.pi * freqs

        result = np.zeros((len(scales), n), dtype=complex)
        for i, a in enumerate(scales):
            # Transformée de Fourier de ψ((t)/a) normalisée : √a · ψ̂(aω)
            psi_fft = (math.sqrt(2 * math.pi * a) *
                       math.pi ** (-0.25) *
                       np.exp(-0.5 * (a * omega - self.omega0) ** 2))
            result[i] = np.fft.ifft(sig_fft * np.conj(psi_fft))
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Rapport de compression
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionReport:
    """
    Métriques de qualité et de compression.

    Taux de compression = taille_originale / taille_compressée
    (en bytes, après compression entropique simulée par zlib).

    PSNR : critère de fidélité.  Au-dessus de 35 dB, la dégradation
    est imperceptible pour l'œil humain.  JPEG typique ≈ 30–40 dB.
    """

    wavelet_name: str
    levels: int
    threshold: float
    threshold_mode: str
    psnr: float
    sparsity: float                # fraction de coefficients nuls
    original_bytes: int
    compressed_bytes: int
    ratio: float = field(init=False)

    def __post_init__(self):
        self.ratio = self.original_bytes / max(self.compressed_bytes, 1)

    def __str__(self) -> str:
        lines = [
            f"╔══ Rapport de compression ══════════════════╗",
            f"║  Ondelette   : {self.wavelet_name:<28}║",
            f"║  Niveaux     : {self.levels:<28}║",
            f"║  Seuil       : {self.threshold:<28.3f}║",
            f"║  Mode        : {self.threshold_mode:<28}║",
            f"║  PSNR        : {self.psnr:<27.2f} dB ║",
            f"║  Sparsité    : {100*self.sparsity:<26.1f} %  ║",
            f"║  Taille orig : {self.original_bytes:<25} o  ║",
            f"║  Taille zlib : {self.compressed_bytes:<25} o  ║",
            f"║  Ratio       : {self.ratio:<27.2f}× ║",
            f"╚════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Codec image
# ─────────────────────────────────────────────────────────────────────────────

class ImageCodec:
    """
    Pipeline complet de compression/décompression d'image par ondelettes.

    Utilisation typique :
        codec = ImageCodec("lenna.jpg", Daubechies4(), levels=4, threshold=20)
        codec.run()
        codec.show()
    """

    def __init__(
        self,
        source: str | Path | np.ndarray,
        wavelet: WaveletBase,
        levels: int = 4,
        threshold: float = 20.0,
        threshold_mode: str = "hard",
    ):
        self.wavelet = wavelet
        self.levels = levels
        self.threshold_value = threshold
        self.threshold_mode = threshold_mode

        # Chargement de l'image
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
            self.original = np.array(img, dtype=np.uint8)
            self.source_name = Path(source).name
        else:
            self.original = source.astype(np.uint8)
            self.source_name = "array"

        self.reconstructed: Optional[np.ndarray] = None
        self.coeffs_per_channel: list[np.ndarray] = []
        self.report: Optional[CompressionReport] = None

    # ------------------------------------------------------------------

    def run(self) -> CompressionReport:
        """Exécute compression + reconstruction, calcule les métriques."""
        channels_out = []
        coeffs_out = []

        for c in range(3):
            plane = self.original[:, :, c]
            coeff, rec = self.wavelet.compress_plane(
                plane, self.levels, self.threshold_value, self.threshold_mode
            )
            channels_out.append(rec)
            coeffs_out.append(coeff)

        self.reconstructed = np.stack(channels_out, axis=2)
        self.coeffs_per_channel = coeffs_out

        # Métriques
        all_coeffs = np.concatenate([c.ravel() for c in coeffs_out])
        sparsity = np.sum(all_coeffs == 0) / all_coeffs.size
        psnr = _psnr(self.original, self.reconstructed)
        orig_bytes = _compressed_size_bytes(self.original.mean(axis=2).astype(np.uint8))
        comp_bytes = sum(
            _compressed_size_bytes((c * 1000).astype(np.int16)) for c in coeffs_out
        ) // 3

        self.report = CompressionReport(
            wavelet_name=self.wavelet.name,
            levels=self.levels,
            threshold=self.threshold_value,
            threshold_mode=self.threshold_mode,
            psnr=psnr,
            sparsity=sparsity,
            original_bytes=orig_bytes,
            compressed_bytes=comp_bytes,
        )
        return self.report

    # ------------------------------------------------------------------

    def show(self, save_path: Optional[str | Path] = None) -> None:
        """Affiche une figure comparative complète (4 panneaux)."""
        if self.reconstructed is None:
            raise RuntimeError("Appeler .run() avant .show()")

        fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            hspace=0.35,
            wspace=0.15,
            left=0.04, right=0.96,
            top=0.88, bottom=0.08,
        )

        title_kw = dict(color="#e8e8e8", fontsize=10, pad=6,
                        fontfamily="monospace")
        label_kw = dict(color="#888888", fontsize=8, family="monospace")

        # --- Image originale ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(self.original)
        ax0.set_title("Original", **title_kw)
        ax0.axis("off")

        # --- Image reconstruite ---
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.reconstructed)
        ax1.set_title(f"Reconstruit  (PSNR = {self.report.psnr:.1f} dB)", **title_kw)
        ax1.axis("off")

        # --- Carte d'erreur absolue ---
        ax2 = fig.add_subplot(gs[0, 2])
        err = np.abs(self.original.astype(int) - self.reconstructed.astype(int)).mean(axis=2)
        im2 = ax2.imshow(err, cmap="inferno", vmin=0, vmax=err.max() or 1)
        ax2.set_title("Erreur absolue (moy. canaux)", **title_kw)
        ax2.axis("off")
        cb = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="intensité")
        cb.ax.yaxis.label.set(**label_kw)

        # --- Coefficients en ondelettes (canal Y-like) ---
        ax3 = fig.add_subplot(gs[1, 0:2])
        luma_coeffs = (
            0.299 * self.coeffs_per_channel[0] +
            0.587 * self.coeffs_per_channel[1] +
            0.114 * self.coeffs_per_channel[2]
        )
        # Normalisation log pour visualiser les ordres de grandeur
        vis = np.log1p(np.abs(luma_coeffs))
        ax3.imshow(vis, cmap="viridis", interpolation="nearest")
        ax3.set_title(
            f"Coefficients ondelettes – {self.wavelet.name}  "
            f"({self.levels} niveaux, sparsité = {100*self.report.sparsity:.1f} %)",
            **title_kw,
        )
        ax3.axis("off")

        # --- Histogramme des coefficients ---
        ax4 = fig.add_subplot(gs[1, 2])
        all_c = luma_coeffs.ravel()
        non_zero = all_c[all_c != 0]
        ax4.hist(non_zero, bins=200, color="#4fc3f7", alpha=0.85, log=True)
        ax4.axvline( self.threshold_value, color="#ff5252", lw=1.5, label=f"+seuil={self.threshold_value}")
        ax4.axvline(-self.threshold_value, color="#ff5252", lw=1.5, label=f"−seuil")
        ax4.set_facecolor("#161616")
        ax4.tick_params(colors="#888888", labelsize=7)
        for sp in ax4.spines.values():
            sp.set_color("#333333")
        ax4.set_title("Distribution des coefficients (log)", **title_kw)
        ax4.set_xlabel("valeur", color="#888888", fontsize=8)
        ax4.set_ylabel("occurrences (log)", color="#888888", fontsize=8)
        ax4.legend(fontsize=7, labelcolor="#cccccc", facecolor="#1e1e1e",
                   edgecolor="#333333")

        # --- Titre global ---
        fig.suptitle(
            f"{self.source_name}  ·  {self.wavelet.name}  ·  "
            f"ratio zlib ≈ {self.report.ratio:.2f}×  ·  "
            f"seuil {self.threshold_mode} = {self.threshold_value}",
            color="#e8e8e8",
            fontsize=13,
            fontfamily="monospace",
            y=0.95,
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.show()
    
    def get_img(self):
        return self.reconstructed

    def get_org(self):
        return self.original
# ─────────────────────────────────────────────────────────────────────────────
#  Comparaison multi-ondelettes
# ─────────────────────────────────────────────────────────────────────────────

def compare_wavelets(
    source: str | Path,
    thresholds: list[float],
    levels: int = 4,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Trace une grille PSNR vs seuil pour Haar et Daubechies4.
    Permet de choisir le bon compromis qualité/compression.
    """
    wavelets = [HaarWavelet(), Daubechies4()]
    results: dict[str, list[tuple[float, float, float]]] = {w.name: [] for w in wavelets}

    img = Image.open(source).convert("RGB")
    original = np.array(img, dtype=np.uint8)

    for wv in wavelets:
        for thr in thresholds:
            codec = ImageCodec(original.copy(), wv, levels=levels, threshold=thr)
            rep = codec.run()
            results[wv.name].append((thr, rep.psnr, rep.ratio))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d0d0d")
    colors = {"Haar": "#ff7043", "Daubechies D4": "#4fc3f7"}

    for wv_name, data in results.items():
        thrs = [d[0] for d in data]
        psnrs = [d[1] for d in data]
        ratios = [d[2] for d in data]
        c = colors[wv_name]
        ax1.plot(thrs, psnrs,  "-o", color=c, ms=4, lw=2, label=wv_name)
        ax2.plot(thrs, ratios, "-o", color=c, ms=4, lw=2, label=wv_name)

    for ax, ylabel, title in [
        (ax1, "PSNR (dB)",       "Qualité en fonction du seuil"),
        (ax2, "Ratio zlib (×)",  "Taux de compression en fonction du seuil"),
    ]:
        ax.set_facecolor("#161616")
        ax.set_xlabel("Seuil", color="#888888", fontsize=9)
        ax.set_ylabel(ylabel,  color="#888888", fontsize=9)
        ax.set_title(title,    color="#e8e8e8", fontsize=10, fontfamily="monospace")
        ax.tick_params(colors="#888888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.legend(fontsize=9, labelcolor="#cccccc", facecolor="#1e1e1e",
                  edgecolor="#333333")
        ax.grid(alpha=0.15, color="#444444")

    fig.suptitle(
        f"Comparaison Haar vs Daubechies D4  ·  {levels} niveaux  ·  {Path(source).name}",
        color="#e8e8e8", fontsize=12, fontfamily="monospace",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


def morlet_scalogram(
    image_path: str | Path,
    row: int = 256,
    scales: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Visualise le scalogramme de Morlet d'une ligne horizontale de l'image.
    Illustre le lien entre la CWT et l'analyse multi-résolution.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=float)
    if row >= arr.shape[0]:
        row = arr.shape[0] // 2
    signal = arr[row, :]

    if scales is None:
        scales = np.geomspace(1, min(len(signal) // 4, 128), 64)

    mw = MorletWavelet(omega0=6.0)
    cwt = mw.cwt(signal, scales, dt=1.0)
    power = np.abs(cwt) ** 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                   facecolor="#0d0d0d",
                                   gridspec_kw={"height_ratios": [1, 2.5]})

    # Signal
    ax1.plot(signal, color="#4fc3f7", lw=0.8)
    ax1.set_facecolor("#161616")
    ax1.set_title(
        f"Ligne {row} de l'image (niveaux de gris)",
        color="#e8e8e8", fontsize=9, fontfamily="monospace",
    )
    ax1.tick_params(colors="#666666", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_color("#333333")

    # Scalogramme
    t_axis = np.arange(len(signal))
    ax2.contourf(t_axis, np.log2(scales), np.log1p(power),
                 levels=50, cmap="inferno")
    ax2.set_facecolor("#161616")
    ax2.set_xlabel("Position (pixels)", color="#888888", fontsize=9)
    ax2.set_ylabel("log₂(échelle)", color="#888888", fontsize=9)
    ax2.set_title(
        "Scalogramme CWT – Morlet (ω₀ = 6)  ·  puissance = |W(a,b)|²",
        color="#e8e8e8", fontsize=10, fontfamily="monospace",
    )
    ax2.tick_params(colors="#666666", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#333333")

    fig.suptitle(
        f"{Path(image_path).name}  ·  Transformée en Ondelettes Continue",
        color="#e8e8e8", fontsize=12, fontfamily="monospace",
        y=0.98,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()



# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée de démonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    IMAGE = sys.argv[1] if len(sys.argv) > 1 else "unnamed1.jpg"

    print("=" * 60)
    print("  Compression par ondelette de Haar")
    print("=" * 60)
    codec_haar = ImageCodec(IMAGE, HaarWavelet(), levels=10, threshold=30, threshold_mode="hard")
    rep_haar = codec_haar.run()
    print(rep_haar)
    codec_haar.show()

    pre = codec_haar.get_org()
    fin = codec_haar.get_img()
    
    plt.subplot(1,2,1)
    plt.imshow(pre)
    plt.subplot(1,2,2)
    plt.imshow(fin)
    plt.show()
    input()

    print("\n" + "=" * 60)
    print("  Compression par Daubechies D4")
    print("=" * 60)
    codec_db4 = ImageCodec(IMAGE, Daubechies4(), levels=4, threshold=15, threshold_mode="soft")
    rep_db4 = codec_db4.run()
    print(rep_db4)
    codec_db4.show()

    print("\n[Comparaison Haar vs DB4 sur plusieurs seuils]")
    compare_wavelets(IMAGE, thresholds=[5, 10, 20, 35, 50, 80, 120], levels=4)

    print("\n[Scalogramme Morlet]")
    morlet_scalogram(IMAGE)