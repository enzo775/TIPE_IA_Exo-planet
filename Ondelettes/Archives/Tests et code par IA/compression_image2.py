"""
wavelet_compression.py
======================

Module de compression d'image par ondelettes.

Deux familles d'ondelettes sont implementees :

1. Haar (ondelette la plus simple, Alfred Haar 1909)
   - Filtres : h0 = [1, 1]/sqrt(2)  (passe-bas)
               h1 = [1,-1]/sqrt(2)  (passe-haut)
   - La DWT revient a des moyennes/differences sur paires de pixels.

2. Daubechies D4 (db2 dans PyWavelets, base de JPEG 2000)
   - 4 coefficients, 2 moments nuls : int t^k psi(t) dt = 0  pour k=0,1
   - Les zones lisses (polynome deg <= 1) => coefficients de detail NULS
     => compression optimale.
   - Implementation vectorisee via np.convolve + extension periodique.

3. Morlet (analyse temps-frequence continue, non utilisee pour la
   compression mais pour visualiser le contenu frequentiel d'une ligne
   de l'image via la Transformee en Ondelettes Continue).

Mesure de compression
---------------------
  a) Sparsite = fraction de coefficients mis a zero par le seuillage.
     C'est ce qu'un codec entropique (CABAC dans JPEG 2000) exploite.
  b) Taille reelle : on sauvegarde le reconstruit en JPEG et on compare
     a la taille PNG de l'original.

Architecture
------------
    WaveletBase     - interface (forward / inverse / threshold)
    ├── HaarWavelet
    └── Daubechies4
    ImageCodec      - pipeline complet
    CompressionReport - metriques
    MorletWavelet   - CWT pour le scalogramme
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


# ---------------------------------------------------------------------------
#  Utilitaires
# ---------------------------------------------------------------------------

def _pad_to_power2(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape
    H = 1 << math.ceil(math.log2(max(h, 2)))
    W = 1 << math.ceil(math.log2(max(w, 2)))
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:h, :w] = arr
    return padded, (h, w)


def _unpad(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return arr[: shape[0], : shape[1]]


def _psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    mse = np.mean((orig.astype(float) - rec.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _jpeg_size(img_array: np.ndarray, quality: int = 85) -> int:
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return buf.tell()


def _png_size(img_array: np.ndarray) -> int:
    buf = io.BytesIO()
    Image.fromarray(img_array.astype(np.uint8)).save(buf, format="PNG")
    return buf.tell()


# ---------------------------------------------------------------------------
#  Classe de base
# ---------------------------------------------------------------------------

class WaveletBase(ABC):
    """
    Interface abstraite pour une ondelette 2-D appliquee canal par canal.

    Transformee 2-D par separabilite (schema de Mallat 1989) :
      - Transformee 1-D sur chaque LIGNE  => sous-bandes L / H
      - Transformee 1-D sur chaque COLONNE => sous-bandes LL / LH / HL / HH

    A chaque niveau n, on ne transforme que le quadrant basse-frequence LL
    (coin superieur-gauche, taille H/2^n x W/2^n).
    """

    name: str = "abstract"

    @abstractmethod
    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        """Transformee directe 1-D (longueur puissance de 2)."""

    @abstractmethod
    def _inverse_1d(self, c: np.ndarray) -> np.ndarray:
        """Transformee inverse 1-D."""

    def _fwd_2d_one(self, plane: np.ndarray) -> np.ndarray:
        out = np.apply_along_axis(self._forward_1d, 1, plane)
        out = np.apply_along_axis(self._forward_1d, 0, out)
        return out

    def _inv_2d_one(self, plane: np.ndarray) -> np.ndarray:
        out = np.apply_along_axis(self._inverse_1d, 0, plane)
        out = np.apply_along_axis(self._inverse_1d, 1, out)
        return out

    def forward(self, plane: np.ndarray, levels: int) -> np.ndarray:
        c = plane.copy().astype(float)
        h, w = c.shape
        for _ in range(levels):
            c[:h, :w] = self._fwd_2d_one(c[:h, :w])
            h //= 2
            w //= 2
        return c

    def inverse(self, c: np.ndarray, levels: int) -> np.ndarray:
        r = c.copy()
        H, W = r.shape
        h = H >> levels
        w = W >> levels
        for _ in range(levels):
            h *= 2
            w *= 2
            r[:h, :w] = self._inv_2d_one(r[:h, :w])
        return r

    def threshold(
        self,
        c: np.ndarray,
        value: float,
        levels: int,
        mode: str = "hard",
    ) -> np.ndarray:
        """
        Seuillage des sous-bandes hautes frequences.
        Le quadrant basse-frequence LL est preserve.

        hard : coeff -> 0 si |coeff| < value
        soft : coeff -> sign(c) * max(|c| - value, 0)
        """
        out = c.copy()
        H, W = out.shape
        lf_h = H >> levels
        lf_w = W >> levels
        mask = np.ones((H, W), dtype=bool)
        mask[:lf_h, :lf_w] = False

        if mode == "hard":
            out[mask & (np.abs(out) < value)] = 0.0
        elif mode == "soft":
            s = np.sign(out)
            out[mask] = s[mask] * np.maximum(np.abs(out[mask]) - value, 0.0)
        else:
            raise ValueError(f"mode inconnu : '{mode}'  (choisir 'hard' ou 'soft')")
        return out

    def compress_plane(
        self,
        plane: np.ndarray,
        levels: int,
        thr: float,
        mode: str = "hard",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (coefficients_seuilles, image_reconstruite) pour un canal."""
        arr, orig_shape = _pad_to_power2(plane.astype(float))
        coeffs   = self.forward(arr, levels)
        coeffs_t = self.threshold(coeffs, thr, levels, mode)
        rec      = self.inverse(coeffs_t, levels)
        rec      = np.clip(_unpad(rec, orig_shape), 0, 255).astype(np.uint8)
        return _unpad(coeffs_t, orig_shape), rec


# ---------------------------------------------------------------------------
#  Haar
# ---------------------------------------------------------------------------

class HaarWavelet(WaveletBase):
    """
    Ondelette de Haar.
    h0 = [1, 1]/sqrt(2)  (passe-bas  => moyenne)
    h1 = [1,-1]/sqrt(2)  (passe-haut => difference)
    Reconstruction parfaite : g0 = h0, g1 = -h1.
    """

    name = "Haar"

    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        s = 1.0 / math.sqrt(2)
        out = np.empty_like(x)
        out[: len(x) // 2] = (x[0::2] + x[1::2]) * s
        out[len(x) // 2 :] = (x[0::2] - x[1::2]) * s
        return out

    def _inverse_1d(self, c: np.ndarray) -> np.ndarray:
        s = 1.0 / math.sqrt(2)
        half = len(c) // 2
        a, d = c[:half], c[half:]
        out = np.empty_like(c)
        out[0::2] = (a + d) * s
        out[1::2] = (a - d) * s
        return out


# ---------------------------------------------------------------------------
#  Daubechies D4  (vectorise, bords periodiques)
# ---------------------------------------------------------------------------

class Daubechies4(WaveletBase):
    """
    Ondelette de Daubechies D4 (db2).

    2 moments nuls => coefficients de detail nuls sur les zones lisses.
    Implementation par convolution vectorisee (np.convolve) avec extension
    periodique aux bords : O(N) au lieu de O(N^2).

    Filtres d'analyse :
        H0 = [(1+s3),(3+s3),(3-s3),(1-s3)] / (4*s2)   passe-bas
        H1 = [-(1-s3),(3-s3),-(3+s3),(1+s3)] / (4*s2)  passe-haut (QMF)

    Filtres de synthese (reconstruction parfaite) :
        G0 = H0 renverse temporellement
        G1 = H1 renverse temporellement

    Reconstruction parfaite garantie par la relation QMF :
        sum_k H0[k] H0[k-2n] = delta[n]   (orthonormalite des translates)
    """

    name = "Daubechies D4"

    _s3 = math.sqrt(3)
    _s2 = math.sqrt(2)

    _H0 = np.array([
        (1 + _s3) / (4 * _s2),
        (3 + _s3) / (4 * _s2),
        (3 - _s3) / (4 * _s2),
        (1 - _s3) / (4 * _s2),
    ])
    _H1 = np.array([
        -(1 - _s3) / (4 * _s2),
         (3 - _s3) / (4 * _s2),
        -(3 + _s3) / (4 * _s2),
         (1 + _s3) / (4 * _s2),
    ])

    _G0 = _H0[::-1].copy()
    _G1 = _H1[::-1].copy()

    def _conv_down(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Convolution periodique puis decimation par 2."""
        L  = len(h)
        xe = np.concatenate([x[-(L - 1):], x])
        y  = np.convolve(xe, h[::-1], mode="valid")
        return y[::2]

    def _conv_up(self, c: np.ndarray, g: np.ndarray, n_out: int) -> np.ndarray:
        """Surelevation par 2 puis convolution periodique."""
        up = np.zeros(len(c) * 2)
        up[::2] = c
        L  = len(g)
        ue = np.concatenate([up[-(L - 1):], up])
        y  = np.convolve(ue, g[::-1], mode="valid")
        return y[:n_out]

    def _forward_1d(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([
            self._conv_down(x, self._H0),
            self._conv_down(x, self._H1),
        ])

    def _inverse_1d(self, c: np.ndarray) -> np.ndarray:
        n    = len(c)
        half = n // 2
        return (self._conv_up(c[:half], self._G0, n)
              + self._conv_up(c[half:], self._G1, n))


# ---------------------------------------------------------------------------
#  Morlet (CWT)
# ---------------------------------------------------------------------------

class MorletWavelet:
    """
    Ondelette de Morlet - non orthogonale, non utilisee pour la compression.

    psi(t) = pi^{-1/4} exp(i*w0*t) exp(-t^2/2)   avec w0=6 (standard)

    CWT calculee dans le domaine de Fourier en O(N log N) :
        W_hat(a, omega) = f_hat(omega) * psi_hat*(a*omega)
    """

    def __init__(self, omega0: float = 6.0):
        self.omega0 = omega0

    def cwt(self, signal: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> np.ndarray:
        n       = len(signal)
        sig_fft = np.fft.fft(signal)
        omega   = 2 * math.pi * np.fft.fftfreq(n, dt)
        result  = np.zeros((len(scales), n), dtype=complex)
        for i, a in enumerate(scales):
            psi_fft = (
                math.sqrt(2 * math.pi * a) * math.pi ** (-0.25)
                * np.exp(-0.5 * (a * omega - self.omega0) ** 2)
            )
            result[i] = np.fft.ifft(sig_fft * np.conj(psi_fft))
        return result


# ---------------------------------------------------------------------------
#  Rapport de compression
# ---------------------------------------------------------------------------

@dataclass
class CompressionReport:
    """
    Metriques de compression et de qualite.

    sparsite       : fraction de coefficients ondelettes mis a zero.
                     C'est CE que compresse le codec entropique (CABAC dans
                     JPEG 2000) : plus c'est eleve, plus le flux est court.

    orig_png_bytes : taille de l'original encode en PNG (sans perte).
    rec_jpeg_bytes : taille du reconstruit encode en JPEG q=85.
    ratio          : orig_png_bytes / rec_jpeg_bytes  (>1 = gain effectif).

    PSNR           : fidelite.  > 35 dB : degradation imperceptible.
                     JPEG standard => 30-40 dB selon le contenu.
    """

    wavelet_name:    str
    levels:          int
    threshold:       float
    threshold_mode:  str
    psnr:            float
    sparsity:        float
    orig_png_bytes:  int
    rec_jpeg_bytes:  int
    ratio:           float = field(init=False)

    def __post_init__(self):
        self.ratio = self.orig_png_bytes / max(self.rec_jpeg_bytes, 1)

    def __str__(self) -> str:
        bar = "=" * 48
        return "\n".join([
            bar,
            f"  Rapport de compression",
            bar,
            f"  Ondelette    : {self.wavelet_name}",
            f"  Niveaux      : {self.levels}",
            f"  Seuil        : {self.threshold:.3f}  ({self.threshold_mode})",
            bar,
            f"  PSNR         : {self.psnr:.2f} dB",
            f"  Sparsite     : {100*self.sparsity:.1f} %",
            bar,
            f"  Original PNG : {self.orig_png_bytes} octets",
            f"  Reconst JPEG : {self.rec_jpeg_bytes} octets",
            f"  Ratio        : {self.ratio:.2f} x",
            bar,
        ])


# ---------------------------------------------------------------------------
#  Codec image
# ---------------------------------------------------------------------------

class ImageCodec:
    """
    Pipeline complet de compression par ondelettes.

    Usage :
        codec = ImageCodec("lenna.jpg", Daubechies4(), levels=4, threshold=15)
        codec.run()
        print(codec.report)
        codec.show()
        codec.save_reconstructed("lenna_db4.jpg")
    """

    def __init__(
        self,
        source:         str | Path | np.ndarray,
        wavelet:        WaveletBase,
        levels:         int   = 4,
        threshold:      float = 15.0,
        threshold_mode: str   = "hard",
    ):
        self.wavelet        = wavelet
        self.levels         = levels
        self.threshold_val  = threshold
        self.threshold_mode = threshold_mode

        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
            self.original    = np.array(img, dtype=np.uint8)
            self.source_name = Path(source).name
        else:
            self.original    = source.astype(np.uint8)
            self.source_name = "array"

        self.reconstructed:      Optional[np.ndarray]        = None
        self.coeffs_per_channel: list[np.ndarray]            = []
        self.report:             Optional[CompressionReport]  = None

    def run(self) -> CompressionReport:
        """Decompose, seuille, reconstruit, calcule les metriques."""
        channels, coeffs = [], []
        for ch in range(3):
            coeff, rec = self.wavelet.compress_plane(
                self.original[:, :, ch],
                self.levels, self.threshold_val, self.threshold_mode,
            )
            channels.append(rec)
            coeffs.append(coeff)

        self.reconstructed      = np.stack(channels, axis=2)
        self.coeffs_per_channel = coeffs

        all_c    = np.concatenate([c.ravel() for c in coeffs])
        sparsity = float(np.sum(all_c == 0) / all_c.size)
        psnr     = _psnr(self.original, self.reconstructed)

        self.report = CompressionReport(
            wavelet_name   = self.wavelet.name,
            levels         = self.levels,
            threshold      = self.threshold_val,
            threshold_mode = self.threshold_mode,
            psnr           = psnr,
            sparsity       = sparsity,
            orig_png_bytes = _png_size(self.original),
            rec_jpeg_bytes = _jpeg_size(self.reconstructed, quality=85),
        )
        return self.report

    def save_reconstructed(self, path: str | Path, jpeg_quality: int = 92) -> Path:
        """
        Sauvegarde l'image reconstruite sur disque.
        Format deduit de l'extension : .jpg/.jpeg => JPEG, sinon => PNG.
        """
        if self.reconstructed is None:
            raise RuntimeError("Appeler .run() avant .save_reconstructed()")
        path = Path(path)
        img  = Image.fromarray(self.reconstructed)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            img.save(path, format="JPEG", quality=jpeg_quality)
        else:
            img.save(path, format="PNG")
        print(f"  Sauvegarde : {path}  ({path.stat().st_size} octets)")
        return path

    def show(self, save_path: Optional[str | Path] = None) -> None:
        """Figure comparative : original / reconstruit / erreur / coefficients / histogramme."""
        if self.reconstructed is None:
            raise RuntimeError("Appeler .run() avant .show()")

        fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
        gs  = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.35, wspace=0.15,
            left=0.04, right=0.96, top=0.88, bottom=0.08,
        )
        tkw = dict(color="#e8e8e8", fontsize=10, pad=6, fontfamily="monospace")

        # Original
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(self.original)
        ax0.set_title("Original", **tkw)
        ax0.axis("off")

        # Reconstruit
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(self.reconstructed)
        ax1.set_title(f"Reconstruit  (PSNR = {self.report.psnr:.1f} dB)", **tkw)
        ax1.axis("off")

        # Erreur absolue
        ax2 = fig.add_subplot(gs[0, 2])
        err = np.abs(self.original.astype(int) - self.reconstructed.astype(int)).mean(axis=2)
        im2 = ax2.imshow(err, cmap="inferno", vmin=0, vmax=max(float(err.max()), 1.0))
        ax2.set_title("Erreur absolue (moy. RGB)", **tkw)
        ax2.axis("off")
        cb = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="intensite")
        cb.ax.yaxis.label.set(color="#888888", fontsize=8)
        cb.ax.tick_params(colors="#888888", labelsize=7)

        # Coefficients (luminance Y)
        ax3 = fig.add_subplot(gs[1, 0:2])
        luma = (
            0.299 * self.coeffs_per_channel[0]
            + 0.587 * self.coeffs_per_channel[1]
            + 0.114 * self.coeffs_per_channel[2]
        )
        ax3.imshow(np.log1p(np.abs(luma)), cmap="viridis", interpolation="nearest")
        ax3.set_title(
            f"Coefficients  {self.wavelet.name}  |  {self.levels} niveaux  "
            f"|  sparsite = {100*self.report.sparsity:.1f}%  "
            f"|  ratio = {self.report.ratio:.2f}x",
            **tkw,
        )
        ax3.axis("off")

        # Annotations sous-bandes LL / LH / HL / HH
        H, W = luma.shape
        lf_h = H >> self.levels
        lf_w = W >> self.levels
        for lbl, x0, y0, dx, dy in [
            ("LL", 0,    0,    lf_w,     lf_h),
            ("LH", lf_w, 0,    W-lf_w,   lf_h),
            ("HL", 0,    lf_h, lf_w,     H-lf_h),
            ("HH", lf_w, lf_h, W-lf_w,   H-lf_h),
        ]:
            ax3.text(
                x0 + dx // 2, y0 + dy // 2, lbl,
                color="white", fontsize=7, ha="center", va="center",
                alpha=0.6, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3, ec="none"),
            )

        # Histogramme
        ax4 = fig.add_subplot(gs[1, 2])
        nz = luma.ravel()
        nz = nz[nz != 0]
        ax4.hist(nz, bins=200, color="#4fc3f7", alpha=0.85, log=True)
        ax4.axvline( self.threshold_val, color="#ff5252", lw=1.5,
                     label=f"+-seuil = {self.threshold_val}")
        ax4.axvline(-self.threshold_val, color="#ff5252", lw=1.5)
        ax4.set_facecolor("#161616")
        ax4.tick_params(colors="#888888", labelsize=7)
        for sp in ax4.spines.values():
            sp.set_color("#333333")
        ax4.set_title("Distribution coefficients (log)", **tkw)
        ax4.set_xlabel("valeur", color="#888888", fontsize=8)
        ax4.set_ylabel("occurrences (log)", color="#888888", fontsize=8)
        ax4.legend(fontsize=7, labelcolor="#cccccc",
                   facecolor="#1e1e1e", edgecolor="#333333")

        fig.suptitle(
            f"{self.source_name}  |  {self.wavelet.name}  |  "
            f"seuil {self.threshold_mode} = {self.threshold_val}  |  "
            f"ratio PNG->JPEG = {self.report.ratio:.2f}x",
            color="#e8e8e8", fontsize=12, fontfamily="monospace", y=0.95,
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


# ---------------------------------------------------------------------------
#  Comparaison multi-ondelettes
# ---------------------------------------------------------------------------

def compare_wavelets(
    source:     str | Path,
    thresholds: list[float],
    levels:     int = 4,
    save_path:  Optional[str | Path] = None,
) -> None:
    """Courbes PSNR, ratio et sparsite en fonction du seuil pour Haar et DB4."""
    wavelets = [HaarWavelet(), Daubechies4()]
    results: dict[str, list] = {w.name: [] for w in wavelets}

    img      = Image.open(source).convert("RGB")
    original = np.array(img, dtype=np.uint8)

    for wv in wavelets:
        for thr in thresholds:
            codec = ImageCodec(original.copy(), wv, levels=levels, threshold=thr)
            rep   = codec.run()
            results[wv.name].append((thr, rep.psnr, rep.ratio, 100 * rep.sparsity))
        print(f"  {wv.name} termine.")

    colors = {"Haar": "#ff7043", "Daubechies D4": "#4fc3f7"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0d0d0d")

    for wv_name, data in results.items():
        thrs     = [d[0] for d in data]
        psnrs    = [d[1] for d in data]
        ratios   = [d[2] for d in data]
        sparsity = [d[3] for d in data]
        c        = colors[wv_name]
        axes[0].plot(thrs, psnrs,    "-o", color=c, ms=4, lw=2, label=wv_name)
        axes[1].plot(thrs, ratios,   "-o", color=c, ms=4, lw=2, label=wv_name)
        axes[2].plot(thrs, sparsity, "-o", color=c, ms=4, lw=2, label=wv_name)

    ylabels = ["PSNR (dB)", "Ratio PNG->JPEG (x)", "Sparsite (%)"]
    titles  = ["Qualite", "Taux de compression reel", "Coefficients nuls"]
    for ax, yl, ti in zip(axes, ylabels, titles):
        ax.set_facecolor("#161616")
        ax.set_xlabel("Seuil", color="#888888", fontsize=9)
        ax.set_ylabel(yl,      color="#888888", fontsize=9)
        ax.set_title(ti, color="#e8e8e8", fontsize=10, fontfamily="monospace")
        ax.tick_params(colors="#888888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.legend(fontsize=9, labelcolor="#cccccc",
                  facecolor="#1e1e1e", edgecolor="#333333")
        ax.grid(alpha=0.15, color="#444444")

    fig.suptitle(
        f"Haar vs Daubechies D4  |  {levels} niveaux  |  {Path(source).name}",
        color="#e8e8e8", fontsize=12, fontfamily="monospace",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


# ---------------------------------------------------------------------------
#  Scalogramme Morlet - avec annotations pedagogiques
# ---------------------------------------------------------------------------

def morlet_scalogram(
    image_path: str | Path,
    row:        int                  = None,
    scales:     Optional[np.ndarray] = None,
    save_path:  Optional[str | Path] = None,
) -> None:
    """
    Visualise le scalogramme de Morlet d'une ligne horizontale de l'image.

    CE QUE TU VOIS
    ──────────────
    Axe horizontal : position dans la ligne (pixels = position b dans W(a,b)).
    Axe vertical   : log2(echelle a).
      - Bas  (petite echelle) = haute frequence = details fins, bords nets.
      - Haut (grande echelle) = basse frequence = structure globale, degrades.
    Couleur : |W(a,b)|^2 (puissance).
      Tache lumineuse = motif fort a cette frequence ET cette position.

    LIEN AVEC LA DWT
    ────────────────
    La DWT (Haar, DB4...) echantillonne ce plan sur les echelles dyadiques
    a = 1, 2, 4, 8... (tirets blancs) avec des ondelettes ORTHOGONALES.
    Les coefficients sont donc decoreles => on peut seuiller chaque
    coefficient independamment sans perturber les autres.
    Les zones SOMBRES = coefficients faibles = ce que le seuillage met a zero.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=float)
    if row is None or row >= arr.shape[0]:
        row = arr.shape[0] // 2
    signal = arr[row, :]

    if scales is None:
        scales = np.geomspace(1, min(len(signal) // 4, 128), 80)

    mw    = MorletWavelet(omega0=6.0)
    cwt   = mw.cwt(signal, scales, dt=1.0)
    power = np.abs(cwt) ** 2

    fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1, 2.5, 0.9],
        hspace=0.45,
        left=0.07, right=0.97, top=0.90, bottom=0.06,
    )
    tkw = dict(color="#e8e8e8", fontsize=9, fontfamily="monospace")

    # Signal brut
    ax_sig = fig.add_subplot(gs[0])
    ax_sig.plot(signal, color="#4fc3f7", lw=0.8)
    ax_sig.set_facecolor("#161616")
    ax_sig.set_title(f"Ligne {row} de l'image (niveaux de gris)", **tkw)
    ax_sig.set_xlim(0, len(signal) - 1)
    ax_sig.tick_params(colors="#666666", labelsize=7)
    for sp in ax_sig.spines.values():
        sp.set_color("#333333")

    # Scalogramme
    t_axis = np.arange(len(signal))
    log_s  = np.log2(scales)
    ax_cwt = fig.add_subplot(gs[1])
    cf = ax_cwt.contourf(t_axis, log_s, np.log1p(power), levels=60, cmap="inferno")
    ax_cwt.set_facecolor("#161616")
    ax_cwt.set_xlabel("Position (pixels)", color="#888888", fontsize=9)
    ax_cwt.set_ylabel("log2(echelle a)  [bas=hautes freq.]", color="#888888", fontsize=9)
    ax_cwt.set_title(
        "Scalogramme Morlet  |W(a,b)|^2 (log)   bas = details fins / haut = structure globale",
        **tkw,
    )
    ax_cwt.tick_params(colors="#666666", labelsize=8)
    for sp in ax_cwt.spines.values():
        sp.set_color("#333333")

    # Lignes dyadiques : echelles echantillonnees par la DWT
    for level in range(1, 8):
        a_dya = 2 ** level
        if scales[0] <= a_dya <= scales[-1]:
            ax_cwt.axhline(math.log2(a_dya), color="#ffffff",
                           lw=0.7, alpha=0.3, ls="--")
            ax_cwt.text(8, math.log2(a_dya) + 0.07,
                        f"DWT niv. {level}  (a={a_dya})",
                        color="white", fontsize=6, alpha=0.55,
                        fontfamily="monospace")

    cb = fig.colorbar(cf, ax=ax_cwt, fraction=0.02, pad=0.01, label="log(1+|W|^2)")
    cb.ax.yaxis.label.set(color="#888888", fontsize=8)
    cb.ax.tick_params(colors="#888888", labelsize=7)

    # Explication textuelle
    ax_txt = fig.add_subplot(gs[2])
    ax_txt.axis("off")
    txt = (
        "Les tirets blancs = echelles dyadiques a=2,4,8... echantillonnees par la DWT.\n"
        "La DWT projette l'image sur des ondelettes ORTHOGONALES a ces echelles exactes :\n"
        "les coefficients sont independants => seuillage sans correlation entre sous-bandes.\n"
        "Zones sombres = coefficients faibles = ce que le seuillage met a zero => compression."
    )
    ax_txt.text(
        0.5, 0.5, txt,
        transform=ax_txt.transAxes,
        ha="center", va="center",
        color="#aaaaaa", fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#161616", ec="#333333"),
    )

    fig.suptitle(
        f"{Path(image_path).name}  |  Transformee en Ondelettes Continue (Morlet, w0=6)",
        color="#e8e8e8", fontsize=12, fontfamily="monospace", y=0.97,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()


# ---------------------------------------------------------------------------
#  Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    IMAGE = sys.argv[1] if len(sys.argv) > 1 else "lenna.jpg"

    # Haar
    print("\n" + "=" * 54)
    print("  Compression Haar")
    print("=" * 54)
    t0 = time.perf_counter()
    codec_haar = ImageCodec(IMAGE, HaarWavelet(), levels=6,
                            threshold=5, threshold_mode="hard")
    rep_haar = codec_haar.run()
    print(f"  Temps : {time.perf_counter() - t0:.2f} s")
    print(rep_haar)
    codec_haar.show()
    codec_haar.save_reconstructed("lenna_haar.jpg")

    pre = codec_haar.get_org()
    fin = codec_haar.get_img()
    
    plt.subplot(1,2,1)
    plt.imshow(pre)
    plt.subplot(1,2,2)
    plt.imshow(fin)
    plt.show()
    input()

    # Daubechies D4
    print("\n" + "=" * 54)
    print("  Compression Daubechies D4")
    print("=" * 54)
    t0 = time.perf_counter()
    codec_db4 = ImageCodec(IMAGE, Daubechies4(), levels=8,
                           threshold=40, threshold_mode="soft")
    rep_db4 = codec_db4.run()
    print(f"  Temps : {time.perf_counter() - t0:.2f} s")
    print(rep_db4)
    codec_db4.show()
    codec_db4.save_reconstructed("lenna_db4.jpg")

    # Comparaison
    print("\n[Comparaison Haar vs Daubechies D4]")
    compare_wavelets(IMAGE, thresholds=[5, 10, 20, 35, 50, 80, 120], levels=4)

    # Scalogramme Morlet
    print("\n[Scalogramme Morlet]")
    morlet_scalogram(IMAGE)