from __future__ import annotations
from importer_image import charger_image

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


class BaseOndelette:
    
    h: np.ndarray
    g: np.ndarray

    def __init__(self, img, levels):
        self.img = img.astype(float) # tableau np
        self.levels = levels # int

    @staticmethod # MIEUX SVP
    def pad(x: np.ndarray, L: int) -> np.ndarray:
        """Padding circulaire : étend x de L-1 échantillons à droite."""
        n = len(x)
        indices = np.arange(n + L - 1) % n   # CORRIGÉ : % n, pas % (n + L - 1)
        return x[indices] 
    
    @staticmethod # MIEUX SVP
    def clip_uint8(img: np.ndarray) -> np.ndarray:
        """Ramène les valeurs dans [0, 255] et convertit en uint8."""
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def transformation_1d(self, x):
        """Banc de filtres passe-bas / passe-haut + sous-échantillonnage ×2."""
        n = len(x)
        L = len(self.h)
        x_pad = self.pad(x, L)
        out = np.zeros(n)

        for i in range(n // 2):
            segment = x_pad[2*i : 2*i + L]
            out[i]          = segment @ self.h   # approximation
            out[i + n // 2] = segment @ self.g   # détail
            # out[i] = np.sum(segment * self.h)

        return out

    def transformation_1d_inverse(self, x: np.ndarray) -> np.ndarray:
        """Reconstruction par sur-échantillonnage + filtres de synthèse."""
        n    = len(x)
        half = n // 2
        L    = len(self.h)
        out  = np.zeros(n)

        approx = x[:half]
        detail = x[half:]

        for i in range(half):
            for k in range(L):
                idx = (2*i + k) % n
                out[idx] += approx[i] * self.h[k] + detail[i] * self.g[k]

        return out
    
    def transformation_2d(self, img):
        n, p = len(img), len(img[0])
        out = np.zeros_like(img)

        for i in range(n):
            out[i, :] = self.transformation_1d(img[i, :])

        for j in range(p):
            out[:, j] = self.transformation_1d(out[:, j])

        return out
    
    def transformation_2d_inverse(self, img):
        n, p = len(img), len(img[0])
        out = np.zeros_like(img)

        for j in range(p):
            out[:, j] = self.transformation_1d_inverse(img[:, j])

        for i in range(n):
            out[i, :] = self.transformation_1d_inverse(out[i, :])
        
        return out

    def forward(self):
        img = self.img
        res = img.copy()
        n, p = len(img), len(img[0])

        for _ in range(self.levels):
            res[:n,:p] = self.transformation_2d(res[:n, :p])
            n //= 2
            p //= 2

        return res
   
    def inverse(self, coeffs, upscale=1):
        n, p = len(coeffs), len(coeffs[0])
        res = np.zeros((upscale*n, upscale*p))
        res[:n,:p] = coeffs

        n //= 2**(self.levels - 1)
        p //= 2**(self.levels - 1)

        for _ in range(self.levels + upscale - 1):
            res[:n, :p] = self.transformation_2d_inverse(res[:n, :p])
            n *= 2
            p *= 2
        return res
    
    def upscale(self):
        img = self.img
        n, p = len(img), len(img[0])
        coeffs  = np.zeros((n*2, p*2))
        est = np.zeros((n, p))
        sud = np.zeros((n, p))
        diag = np.zeros((n, p))
        for i in range(1, n - 1):
            for j in range(1, p - 1):
                est[i][j] = (img[i][j+1] - img[i][j-1])/4
                sud[i][j] = (img[i+1][j] - img[i-1][j])/4
                diag[i][j] = (img[i+1][j+1] - img[i][j])/8
        
        coeffs[:n, :p] =  2*img
        coeffs[:n, p:] =  0.6 * est
        coeffs[n:, :p] =  0.6 * sud
        coeffs[n:, p:] =  0.4 * diag
        
    
        res1 = self.inverse(coeffs, upscale=1)
        res = np.zeros((n*2, p*2))
        
        for i in range(1, 2*n - 1):
            for j in range(1, 2*p - 1):
                res[i][j] = (res1[i][j] + res1[i][j+1] + res1[i+1][j] + res1[i+1][j+1] )/4

        return res
    
    def upscale2(self, image) -> np.ndarray:
        """
        Upscale ×2 par projection dans l'espace V_{j+1}.

        Principe (AMR) : l'image basse résolution appartient à V_j.
        Sa projection dans V_{j+1} s'obtient en la plaçant dans le bloc
        LL d'un buffer 2×2 fois plus grand (sous-bandes LH, HL, HH = 0)
        et en appliquant une passe de DWT inverse.

        Pour Haar : chaque pixel I[i,j] génère un bloc 2×2 de valeur
        constante I[i,j]/2, ce qui est l'interpolation la plus régulière
        compatible avec la norme L² et la structure d'échelle de Haar.
        """
        n, p = image.shape
        buf = np.zeros((2 * n, 2 * p))
        buf[:n, :p] = image     # image dans LL, hauts détails = 0

        # On crée une instance temporaire avec la bonne taille pour accéder à t2d_inv
        w_up = self.__class__(np.zeros((2 * n, 2 * p)), levels=1)
        result = w_up.inverse(buf)

        # Renormalisation : compense le facteur 1/2 introduit par Haar
        # (optionnel selon si on veut conserver l'énergie ou les valeurs)
        return result * 2.0
        
    @staticmethod
    def seuillage(
        coeffs: np.ndarray,
        seuil: float,
        mode: str = "dur",
    ) -> np.ndarray:
        c = coeffs.copy()
        abs_c = np.abs(c)

        if mode == "dur":
            return np.where(abs_c >= seuil, c, 0.0)

        elif mode == "doux":
            return np.sign(c) * np.maximum(abs_c - seuil, 0.0)

        elif mode == "exp":
            # Transition continue : f(seuil) = 0, f(+inf) -> |c| - seuil
            # Paramètre alpha : contrôle la rapidité de la transition
            # alpha grand -> proche du seuillage dur ; alpha petit -> proche du doux
            alpha = 1
            excess = np.maximum(abs_c - seuil, 0.0)         # 0 en dessous du seuil
            enveloppe = excess * (1.0 - np.exp(-alpha * excess))
            return np.sign(c) * enveloppe

        else:
            raise ValueError(f"Mode inconnu : {mode!r}. Choisir 'dur', 'doux' ou 'exp'.")


    def taux_compression(self, coeffs: np.ndarray, seuil: float) -> float:
        """
        Proportion de coefficients qui seront mis à zéro après seuillage.
        """
        n_zeros = np.sum(np.abs(coeffs) < seuil)   # bug corrigé : abs()
        return n_zeros / coeffs.size               # .size plus propre que n*p
    
    def psnr(self, rec):
        """Peak Signal-to-Noise Ratio en dB. > 40 dB : excellent. < 30 dB : visible."""
        orig = self.img
        mse = float(np.mean((orig.astype(float) - rec.astype(float)) ** 2))
        return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)

    def afficher(
        self,
        coeffs: np.ndarray,
        reconstruit: np.ndarray,
        seuil: float | None = None,
    ) -> None:
        pass
    
class Haar(BaseOndelette):
    def __init__(self, img, levels):
        super().__init__(img, levels)
        s2 = np.sqrt(2)
        # Coefficients orthogonaux (indispensables pour l'inverse) + energie wtf ??
        self.h = np.array([ 1/s2,  1/s2])
        self.g = np.array([ 1/s2, -1/s2])


##### CODE TEST

def comparer_seuils(wavelet, seuil):
    coeffs = wavelet.forward()
    
    coeffs1 = wavelet.seuillage(coeffs, seuil, mode="dur")
    coeffs2= wavelet.seuillage(coeffs, seuil, mode="doux")
    coeffs3 = wavelet.seuillage(coeffs, seuil, mode="exp")

    reconstructed1 = wavelet.inverse(coeffs1)
    reconstructed1 = BaseOndelette.clip_uint8(reconstructed1)

    reconstructed2 = wavelet.inverse(coeffs2)
    reconstructed2 = BaseOndelette.clip_uint8(reconstructed2)

    reconstructed3 = wavelet.inverse(coeffs3)
    reconstructed3 = BaseOndelette.clip_uint8(reconstructed3)

    coeffs = BaseOndelette.clip_uint8(coeffs)

    # Affichage Matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(reconstructed1, cmap='gray')
    plt.title("Seuillage dur")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed2, cmap='gray')
    plt.title("Seuillage doux")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed3, cmap='gray')
    plt.title("Seuillage exp")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def graphe_seuils(wavelet, min, max, n, mode_seuillage):
    # Compression (Forward)
    coeffs = wavelet.forward()
    step = (max - min) // n
    seuils = list(range(min, max, step))
    psnr = []
    tau = []
    for seuil in seuils:
        tau.append(100*wavelet.taux_compression(coeffs, seuil))
        coeffs_courant = wavelet.seuillage(coeffs, seuil, mode=mode_seuillage)
        r = wavelet.inverse(coeffs_courant)
        r = BaseOndelette.clip_uint8(r)
        psnr.append(wavelet.psnr(r))

    # Création de la figure avec 3 sous-graphiques
    plt.figure(figsize=(15, 5))

    # Premier sous-graphe : tau en fonction du seuil
    plt.subplot(1, 3, 1)
    plt.plot(seuils, tau, marker='o')
    plt.xlabel("Seuil")
    plt.ylabel("Taux de compression (%)")
    plt.title("Taux de compression en fonction du seuil")
    plt.grid(True)

    # Deuxième sous-graphe : PSNR en fonction du seuil
    plt.subplot(1, 3, 2)
    plt.plot(seuils, psnr, marker='o')
    plt.axhline(y=40, color='g', linestyle='--', label='Excellente')
    plt.axhline(y=35, color='y', linestyle='--', label='Imperceptible')
    plt.axhline(y=30, color='r', linestyle='--', label='Visible')
    plt.xlabel("Seuil")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR en fonction du seuil")
    plt.grid(True)
    plt.legend()

    # Troisième sous-graphe : PSNR en fonction de tau
    plt.subplot(1, 3, 3)
    plt.plot(tau, psnr, marker='o')
    plt.axhline(y=40, color='g', linestyle='--', label='Excellente')
    plt.axhline(y=35, color='y', linestyle='--', label='Imperceptible')
    plt.axhline(y=30, color='r', linestyle='--', label='Visible')
    plt.xlabel("Taux de compression (%)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR en fonction du taux de compression")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def show(wavelet, seuil, mode_seuillage):
    # Compression (Forward)
    coeffs = wavelet.forward()

    tau = 100*wavelet.taux_compression(coeffs, seuil)
    print("Image compressée à %.2f %%" % (tau))
    
    coeffs = wavelet.seuillage(coeffs, seuil, mode=mode_seuillage)

    r = wavelet.inverse(coeffs)
    r = BaseOndelette.clip_uint8(r)

    coeffs = BaseOndelette.clip_uint8(coeffs)

    # 3. Affichage Matplotlib
    plt.figure(figsize=(15, 5))

    # Image Originale
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Originale")
    plt.axis('off')

    # Image des coefficients (on utilise log pour mieux voir les détails)
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(coeffs), cmap='gray', vmax=np.percentile(np.abs(coeffs), 95))
    plt.title(f"Coefficients Haar ({levels} niveaux)")
    plt.axis('off')

    # Image Reconstruite
    plt.subplot(1, 3, 3)
    plt.imshow(r, cmap='gray')
    plt.title("Décompressée (Inverse)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def distribution_coeffs(wavelet, echelle_log=False):
    img = wavelet.forward()
    data = img.ravel()
    bins= len(data)//50

    vmin = data.min()
    vmax = data.max()

    plt.figure(figsize=(10,5))
    plt.hist(
        data,
        bins=bins,
        range=(vmin, vmax),
        log=echelle_log
    )

    plt.xlabel("Valeur du coefficient")
    plt.ylabel("Occurrences")
    plt.title(
        f"Distribution des coefficients\n"
        f"min={vmin:.2f}, max={vmax:.2f}"
    )
    plt.grid(alpha=0.3)

    plt.show()

if __name__ == "__main__":
    image = "barbara.jpg"
    try:
        # Charge en niveaux de gris et redimensionne en puissance de 2 pour Haar
        img_raw = charger_image(image, gray=True) # "L" pour n&b
        img_np = np.array(img_raw, dtype=np.uint8)
        
    except FileNotFoundError:
        print("Fichier %s non trouvé, génération d'un pattern de test." % image)
        img_np = (np.indices((512, 512)).sum(axis=0) % 255)

    # 2. Traitement
    levels = 3
    wavelet = Haar(img_np, levels)
    graphe_seuils(wavelet, 5, 200, 30, 'doux')
    #comparer_seuils(wavelet, 60)
    #show(wavelet, 200, "doux")
    #distribution_coeffs(wavelet, echelle_log=True)
    