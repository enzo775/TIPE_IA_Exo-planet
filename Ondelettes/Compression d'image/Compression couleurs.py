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


class BaseOndeletteCouleur:
    
    h: np.ndarray
    g: np.ndarray
    h_synth : np.ndarray
    g_synth : np.ndarray

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
                out[idx] += approx[i] * self.h_synth[k] + detail[i] * self.g_synth[k]

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
        n1, p1 = len(img), len(img[0])
        for i in range(3):
            n, p = n1, p1
            for _ in range(self.levels):
                res[:n,:p, i] = self.transformation_2d(res[:n, :p, i])
                n //= 2
                p //= 2

        return res
    
    def inverse(self, coeffs, upscale=1):
        n1, p1 = len(coeffs), len(coeffs[0])
        res = np.zeros((upscale*n1, upscale*p1, 3))
        res[:n1,:p1, :] = coeffs
        for i in range(3):
            n = n1 // ( 2**(self.levels - 1) )
            p = p1 // ( 2**(self.levels - 1) )

            for _ in range(self.levels + upscale - 1):
                res[:n, :p, i] = self.transformation_2d_inverse(res[:n, :p, i])
                n *= 2
                p *= 2
        return res
    
    def taux_compression(self, coeffs, seuil, canal="RGB"):
        """
        Proportion de coefficients mis à zéro après seuillage.
        """
        #n_zeros = np.sum(np.abs(coeffs) < seuil)   # bug corrigé : abs()
        #return n_zeros / coeffs.size
        n, p, _ = coeffs.shape

        ll_h = n // (2 ** (self.levels - 1))
        ll_w = p // (2 ** (self.levels - 1))

        # seuil par canal
        thresholds = np.full(coeffs.shape, seuil, dtype=float)

        if canal.upper() == "YCC":
            thresholds[:, :, 0] = seuil / 2
            thresholds[:, :, 1] = 2*seuil
            thresholds[:, :, 2] = 2*seuil


        # masque : seuls les détails peuvent être annulés
        mask = np.ones(coeffs.shape, dtype=bool)
        mask[:ll_h, :ll_w, :] = False

        n_zeros = np.sum(mask & (np.abs(coeffs) < thresholds))
        n_total = np.sum(mask)

        return n_zeros / n_total
    
    @staticmethod
    def taux_compression_maison(coeffs, seuil):
        """
        Proportion de coefficients mis à zéro après seuillage.
        """
        n_zeros = 0   # bug corrigé : abs()
        for i in range(len(coeffs)):
            for j in range(len(coeffs[0])):
                for k in range(3):
                    if np.abs(coeffs[i,j,k]) < seuil:
                        n_zeros += 1
        return n_zeros / (len(coeffs)*len(coeffs[0])*3)
    
    def seuillage(
        self,
        coeffs: np.ndarray,
        seuil: float,
        mode: str = "dur",
        canal: str = "RGB"
    ) -> np.ndarray:
        n, p, _ = coeffs.shape
        ll_h = n // (2**(self.levels - 1))
        ll_w = p // (2**(self.levels - 1))
        mask = np.ones_like(coeffs, dtype=bool)

        mask[:ll_h, :ll_w, :] = False
        
        c = coeffs.copy()
        abs_c = np.abs(c)

        thresholds = np.full(coeffs.shape, seuil)
        if canal.upper() == "YCC":
            thresholds[:, :, 0] = seuil / 1.5
            thresholds[:, :, 1] = 1.5*seuil
            thresholds[:, :, 2] = 1.5*seuil

        if mode == "dur":
            #return np.where(abs_c >= seuil, c, 0.0)
            out = c.copy()

            to_zero = mask & (abs_c < thresholds)

            out[to_zero] = 0.0

            return out

        elif mode == "doux":
            # return np.sign(c) * np.maximum(abs_c - seuil, 0.0)

            out = np.sign(c) * np.maximum(abs_c - thresholds, 0.0)

            # restauration exacte du LL
            out[:ll_h, :ll_w, :] = c[:ll_h, :ll_w, :]

            return out

        elif mode == "exp":
            # Transition continue : f(seuil) = 0, f(+inf) -> |c| - seuil
            # Paramètre alpha : contrôle la rapidité de la transition
            # alpha grand -> proche du seuillage dur ; alpha petit -> proche du doux
            #alpha = 10
            #excess = np.maximum(abs_c - seuil, 0.0)         # 0 en dessous du seuil
            #enveloppe = excess * (1.0 - np.exp(-alpha * excess))
            #return np.sign(c) * enveloppe

            alpha = 10

            excess = np.maximum(abs_c - thresholds, 0.0)

            out = np.sign(c) * excess * (1 - np.exp(-alpha * excess))

            # LL non seuillé
            out[:ll_h, :ll_w, :] = c[:ll_h, :ll_w, :]

            return out

        else:
            raise ValueError(f"Mode inconnu : {mode!r}. Choisir 'dur', 'doux' ou 'exp'.")
        
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
    
class Haar(BaseOndeletteCouleur):

    def __init__(self, img, levels):
        super().__init__(img, levels)

        s2 = np.sqrt(2)

        self.h = np.array([
            1/s2,
            1/s2
        ])

        self.g = np.array([
            1/s2,
           -1/s2
        ])

        # Haar orthogonale
        self.h_synth = self.h.copy()
        self.g_synth = self.g.copy()

class Daubechies(BaseOndeletteCouleur):

    def __init__(self, img, levels):
        super().__init__(img, levels)

        sqrt3 = np.sqrt(3)
        denom = 4*np.sqrt(2)

        h0 = (1 + sqrt3)/denom
        h1 = (3 + sqrt3)/denom
        h2 = (3 - sqrt3)/denom
        h3 = (1 - sqrt3)/denom

        # analyse
        self.h = np.array([
            h0,
            h1,
            h2,
            h3
        ])

        self.g = np.array([
            h3,
           -h2,
            h1,
           -h0
        ])

        self.h_synth = self.h.copy()
        self.g_synth = self.g.copy()

class Daubechies3(BaseOndeletteCouleur):

    def __init__(self, img, levels):
        super().__init__(img, levels)

        self.h = np.array([
            0.3326705529500826,
            0.8068915093110928,
            0.4598775021184915,
           -0.1350110200102546,
           -0.0854412738820267,
            0.0352262918857095
        ])

        self.g = np.array([
            0.0352262918857095,
            0.0854412738820267,
           -0.1350110200102546,
           -0.4598775021184915,
            0.8068915093110928,
           -0.3326705529500826
        ])

        self.h_synth = self.h.copy()
        self.g_synth = self.g.copy()

class Daubechies4(BaseOndeletteCouleur):

    def __init__(self, img, levels):
        super().__init__(img, levels)

        self.h = np.array([
            0.2303778133088964,
            0.7148465705529154,
            0.6308807679298587,
           -0.0279837694168599,
           -0.1870348117188811,
            0.0308413818355607,
            0.0328830116668852,
           -0.0105974017850690
        ])

        self.g = np.array([
           -0.0105974017850690,
           -0.0328830116668852,
            0.0308413818355607,
            0.1870348117188811,
           -0.0279837694168599,
           -0.6308807679298587,
            0.7148465705529154,
           -0.2303778133088964
        ])

        self.h_synth = self.h.copy()
        self.g_synth = self.g.copy()

class Daubechies5(BaseOndeletteCouleur):

    def __init__(self, img, levels):
        super().__init__(img, levels)

        self.h = np.array([
            0.1601023979741929,
            0.6038292697971895,
            0.7243085284377729,
            0.1384281459013203,
           -0.2422948870663823,
           -0.0322448695850295,
            0.0775714938400459,
           -0.0062414902127983,
           -0.0125807519990820,
            0.0033357252854738
        ])

        self.g = np.array([
            0.0033357252854738,
            0.0125807519990820,
           -0.0062414902127983,
           -0.0775714938400459,
           -0.0322448695850295,
            0.2422948870663823,
            0.1384281459013203,
           -0.7243085284377729,
            0.6038292697971895,
           -0.1601023979741929
        ])

        self.h_synth = self.h.copy()
        self.g_synth = self.g.copy()

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

def ycc_to_rgb(ycc_array):
    Y  = ycc_array[:, :, 0].astype(float)
    Cr = ycc_array[:, :, 1].astype(float)
    Cb = ycc_array[:, :, 2].astype(float)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.714136 * (Cr - 128) - 0.344136 * (Cb - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb_array = np.stack((R, G, B), axis=-1)

    # retour dans l'intervalle valide
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

    return rgb_array

def comparer_seuils(wavelet, seuil):
    coeffs = wavelet.forward()
    
    coeffs1 = wavelet.seuillage(coeffs, seuil, mode="dur")
    coeffs2= wavelet.seuillage(coeffs, seuil, mode="doux")
    coeffs3 = wavelet.seuillage(coeffs, seuil, mode="exp")

    reconstructed1 = wavelet.inverse(coeffs1)
    reconstructed1 = BaseOndeletteCouleur.clip_uint8(reconstructed1)

    reconstructed2 = wavelet.inverse(coeffs2)
    reconstructed2 = BaseOndeletteCouleur.clip_uint8(reconstructed2)

    reconstructed3 = wavelet.inverse(coeffs3)
    reconstructed3 = BaseOndeletteCouleur.clip_uint8(reconstructed3)

    coeffs = BaseOndeletteCouleur.clip_uint8(coeffs)

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
        r = BaseOndeletteCouleur.clip_uint8(r)
        psnr.append(wavelet.psnr(r))

    # Création de la figure avec 3 sous-graphiques
    plt.figure(figsize=(15, 5))

    # Premier sous-graphe : tau en fonction du seuil
    plt.subplot(1, 3, 1)
    plt.plot(seuils, tau, marker='.')
    plt.xlabel("Seuil")
    plt.ylabel("Taux de compression (%)")
    plt.title("Taux de compression en fonction du seuil")
    plt.grid(True)

    # Deuxième sous-graphe : PSNR en fonction du seuil
    plt.subplot(1, 3, 2)
    plt.plot(seuils, psnr, marker='.')
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
    plt.plot(tau, psnr, marker='.')
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
    r = BaseOndeletteCouleur.clip_uint8(r)

    coeffs = BaseOndeletteCouleur.clip_uint8(coeffs)

    # 3. Affichage Matplotlib
    plt.figure(figsize=(15, 5))

    # Image Originale
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Originale")
    plt.axis('off')

    # Image des coefficients (on utilise log pour mieux voir les détails)
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(coeffs), vmax=np.percentile(np.abs(coeffs), 95))
    plt.title(f"Coefficients Haar ({levels} niveaux)")
    plt.axis('off')

    # Image Reconstruite
    plt.subplot(1, 3, 3)
    plt.imshow(r)
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
        f"Distribution des coefficients"
    )
    plt.axvline(x=-40, color='r', linestyle='--')
    plt.axvline(x=40, color='r', linestyle='--')
    plt.grid(alpha=0.3)

    plt.show()

def rgb_vs_ycc(img, levels, seuil=10, mode_seuillage="dur", ondelette="HAAR"):
    if ondelette.upper() == "HAAR":
        w1 = Haar(img, levels)
        w2 = Haar(rgb_to_ycc(img), levels)
    elif ondelette.upper() == "DAUBECHIES":
        w1 = Daubechies(img, levels)
        w2 = Daubechies(rgb_to_ycc(img), levels)

    c1 = w1.forward()
    c2 = w2.forward()

    t1, t2 = w1.taux_compression(c1, seuil, "RGB"), w1.taux_compression(c1, seuil, "YCC")
    c1 = w1.seuillage(c1, seuil, mode=mode_seuillage, canal="RGB") 
    c2 = w2.seuillage(c2, seuil, mode=mode_seuillage, canal="YCC")

    r1 = w1.inverse(c1)
    r1 = BaseOndeletteCouleur.clip_uint8(r1)
    
    r2 = w2.inverse(c2)
    r2 = BaseOndeletteCouleur.clip_uint8(ycc_to_rgb(r2))

    # 3. Affichage Matplotlib
    plt.figure(figsize=(10, 5))

    # Image Originale
    plt.subplot(1, 2, 1)
    plt.imshow(r1)
    plt.title("RGB")
    plt.axis('off')

    # Image Reconstruite
    plt.subplot(1, 2, 2)
    plt.imshow(r2)
    plt.title("YCC")
    plt.axis('off')

    # Texte sous chaque sous-graphe
    plt.figtext(0.25, 0.05, f"Taux de compression : {t1:.3f}", ha='center', fontsize=12)
    plt.figtext(0.75, 0.05, f"Taux de compression : {t2:.3f}", ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def labo(img, levels, mode_seuillage, min, max, n):
    # Initialisation des wavelets
    wavelets = {
        "Haar": Haar(img, levels),
        "Daubechies": Daubechies(img, levels),
        "Daubechies3": Daubechies3(img, levels),
        "Daubechies4": Daubechies4(img, levels),
        "Daubechies5": Daubechies5(img, levels)
    }

    # Calcul des seuils
    step = (max - min) // n
    seuils = list(range(min, max, step))

    # Dictionnaire pour stocker les résultats (tau et psnr pour chaque wavelet)
    results = {name: {"tau": [], "psnr": []} for name in wavelets}

    # Boucle sur chaque wavelet
    for name, wavelet in wavelets.items():
        coeffs = wavelet.forward()
        for s in seuils:
            tau = 100 * wavelet.taux_compression(coeffs, s)
            coeffs_seuilles = wavelet.seuillage(coeffs, s, mode=mode_seuillage)
            r = wavelet.inverse(coeffs_seuilles)
            r = BaseOndeletteCouleur.clip_uint8(r)
            psnr = wavelet.psnr(r)
            results[name]["tau"].append(tau)
            results[name]["psnr"].append(psnr)

    # Création de la figure avec 3 sous-graphiques
    plt.figure(figsize=(18, 5))

    # --- 1. Tau en fonction du seuil ---
    plt.subplot(1, 3, 1)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, (name, data) in enumerate(results.items()):
        plt.plot(seuils, data["tau"], marker='.', color=colors[i % len(colors)], label=name)
    plt.xlabel("Seuil")
    plt.ylabel("Taux de compression (%)")
    plt.title("Taux de compression en fonction du seuil")
    plt.grid(True)
    plt.legend()

    # --- 2. PSNR en fonction du seuil ---
    plt.subplot(1, 3, 2)
    for i, (name, data) in enumerate(results.items()):
        plt.plot(seuils, data["psnr"], marker='.', color=colors[i % len(colors)], label=name)
    plt.axhline(y=40, color='g', linestyle='--', label='Excellente')
    plt.axhline(y=35, color='y', linestyle='--', label='Imperceptible')
    plt.axhline(y=30, color='r', linestyle='--', label='Visible')
    plt.xlabel("Seuil")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR en fonction du seuil")
    plt.grid(True)
    plt.legend()

    # --- 3. PSNR en fonction de tau ---
    plt.subplot(1, 3, 3)
    for i, (name, data) in enumerate(results.items()):
        plt.plot(data["tau"], data["psnr"], marker='.', color=colors[i % len(colors)], label=name)
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


if __name__ == "__main__":
    image = "pernette.jpg"
    try:
        img_raw = charger_image(image, gray=False) # "L" pour n&b
        img_np = np.array(img_raw, dtype=np.uint8)
        # img_np = rgb_to_ycc(img_np)

    except FileNotFoundError:
        print("Fichier %s non trouvé, génération d'un pattern de test." % (image))
        img_np = (np.indices((512, 512, 3)).sum(axis=0) % 255)

    # 2. Traitement
    levels = 3
    wavelet = Haar(img_np, levels)
    
    m = "dur"
    seuil = 35

    #graphe_seuils(wavelet, 5, 150, 40, 'dur')
    #comparer_seuils(wavelet, seuil)
    show(wavelet, seuil, m)
    #distribution_coeffs(wavelet, echelle_log=True)
    #rgb_vs_ycc(img_np, levels, seuil, m, "HAAR")
    #labo(img_np, levels, m, 5, 150, 30)