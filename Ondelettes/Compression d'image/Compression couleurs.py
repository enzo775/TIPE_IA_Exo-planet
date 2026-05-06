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
    
    def inverse(self, coeffs):
        n, p = len(coeffs), len(coeffs[0])
        n //= 2**(self.levels - 1)
        p //= 2**(self.levels - 1) # ARRRGFGDJGDHFDFDHJJDNYFUHGOIEUYJDFGHO7IDFBHU

        res = coeffs.copy()
        for _ in range(self.levels):
            res[:n, :p] = self.transformation_2d_inverse(res[:n, :p])
            n *= 2
            p *= 2
        return res
    
    def taux_compression(self, coeffs, seuil):
        """
        Proportion de coefficients mis à zéro après seuillage.
        """
        naze = np.sum(np.abs(coeffs) < seuil)
        return naze / coeffs.size
    
    @staticmethod
    def seuillage(
        coeffs: np.ndarray,
        seuil: float,
        mode: str = "dur",
    ) -> np.ndarray:
        """
        Seuillage des coefficients ondelettes.

        Paramètres
        ----------
        coeffs : tableau de coefficients (sortie de forward)
        seuil  : valeur de seuil λ ≥ 0
        mode   : "dur"  → met à 0 si |c| < λ, garde sinon
                 "doux" → met à 0 si |c| < λ, réduit vers 0 sinon
                          (c → sign(c) * (|c| - λ))

        Retourne ?
        --------
        Tableau de coefficients seuillés (même forme que coeffs).
        """
        pass

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

def rgb_to_ycc(rgb_array):
    R = rgb_array[:, :, 0].astype(float)
    G = rgb_array[:, :, 1].astype(float)
    B = rgb_array[:, :, 2].astype(float)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = 128 + 0.5 * R - 0.4187 * G - 0.0813 * B
    Cb = 128 - 0.1687 * R - 0.3313 * G + 0.5 * B

    ycc_array = np.stack((Y, Cr, Cb), axis=-1)
    return ycc_array

if __name__ == "__main__":
    image = "rose.jpg"
    try:
        # Charge en niveaux de gris et redimensionne en puissance de 2 pour Haar
        img_raw = Image.open(image).convert("RGB") # "L" pour n&b
        img_np = np.array(img_raw, dtype=np.uint8)
        r = img_np[:,:,0]
        g = img_np[:,:,1]
        b = img_np[:,:,2]

    except FileNotFoundError:
        print("Fichier lenna.jpg non trouvé, génération d'un pattern de test.")
        img_np = (np.indices((512, 512)).sum(axis=0) % 255)

    # 2. Traitement
    levels = 3
    #wavelet = Haar(img_np, levels)

    w1 = Haar(r, levels)
    w2 = Haar(g, levels)
    w3 = Haar(b, levels)
    print("a")
    # Compression (Forward)
    #coeffs = wavelet.forward()

    c1 = w1.forward()
    c2 = w2.forward()
    c3 = w3.forward()
    n = len(c1)
    c1[n//2 :, n//2: ] = 0
    c1[: n//2, n//2: ] = 0 
    c1[n//2 :, : n//2] = 0
    c2[n//2 :, n//2: ] = 0
    c2[: n//2, n//2: ] = 0 
    c2[n//2 :, : n//2] = 0
    c3[n//2 :, n//2: ] = 0
    c3[: n//2, n//2: ] = 0 
    c3[n//2 :, : n//2] = 0
    coeffs = np.stack((c1,c2,c3), axis=-1)
    print("b")
    # Décompression (Inverse)
    #reconstructed = wavelet.inverse(coeffs)

    r1 = w1.inverse(c1)
    r2 = w2.inverse(c2)
    r3 = w3.inverse(c3)
    reconstructed = np.stack((r1,r2,r3), axis=-1)
    reconstructed = BaseOndelette.clip_uint8(reconstructed)
    coeffs = BaseOndelette.clip_uint8(coeffs)

    # 3. Affichage Matplotlib
    plt.figure(figsize=(15, 5))

    # Image Originale
    plt.subplot(1, 3, 1)
    #plt.imshow(img_np, cmap='gray')
    plt.imshow(img_np)
    plt.title("Originale")
    plt.axis('off')

    # Image des coefficients (on utilise log pour mieux voir les détails)
    plt.subplot(1, 3, 2)
    #plt.imshow(np.abs(coeffs), cmap='gray', vmax=np.percentile(np.abs(coeffs), 95))
    plt.imshow(np.abs(coeffs), vmax=np.percentile(np.abs(coeffs), 95))
    plt.title(f"Coefficients Haar ({levels} niveaux)")
    plt.axis('off')

    # Image Reconstruite
    plt.subplot(1, 3, 3)
    #plt.imshow(reconstructed, cmap='gray')
    plt.imshow(reconstructed)
    plt.title("Décompressée (Inverse)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()