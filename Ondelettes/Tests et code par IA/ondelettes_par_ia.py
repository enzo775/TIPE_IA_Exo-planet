import math
import matplotlib.pyplot as plt
import numpy as np

# --- 1. BIBLIOTHÈQUE D'ONDELETTES (Natif) ---

def haar(t):
    if 0 <= t < 0.5: return 1.0
    if 0.5 <= t <= 1: return -1.0
    return 0.0

def ricker(t):
    t2 = t**2
    return (1.0 - t2) * math.exp(-t2 / 2.0)

def morlet(t):
    return math.exp(-t**2 / 2.0) * math.cos(5.0 * t)

def morlet2(t):
    return 2*math.sin(np.pi*t/2)*math.cos(3*np.pi*t/2)/(np.pi*(t+0.01))
# --- 2. CONFIGURATION : CHANGEZ L'ONDELETTE ICI ---

# Options possibles : haar, ricker, morlet
CHOIX_ONDELETTE = haar

def f(x):
    """Votre fonction personnalisée"""
    # Exemple : un saut de fréquence
    if x < 0.3:
        return math.sin(2 * math.pi * 10 * x)
    elif x < 0.6:
        return math.sin(2 * math.pi * 20 * x)
    return math.cosh(x)*math.cos(x)

# --- 3. PARAMÈTRES ---

N = 150                 # Résolution temporelle (augmenter pour plus de précision)
x_vals = [i/N for i in range(N)]
dx = 1.0 / N
scales = [0.02 + i*0.01 for i in range(25)] # Paramètres 'a'

# --- 4. CALCULS (ANALYSE & BIJECTION) ---

# A. Analyse (Transformée)
coeffs = np.zeros((len(scales), N))
for i, a in enumerate(scales):
    for j, b in enumerate(x_vals):
        somme = 0
        for k in range(N-1):
            # Méthode des trapèzes pour l'intégrale
            val1 = f(x_vals[k]) * (1/math.sqrt(a)) * CHOIX_ONDELETTE((x_vals[k]-b)/a)
            val2 = f(x_vals[k+1]) * (1/math.sqrt(a)) * CHOIX_ONDELETTE((x_vals[k+1]-b)/a)
            somme += (val1 + val2) * dx / 2.0
        coeffs[i, j] = somme

# B. Synthèse (Reconstruction / Bijection)
f_rec = [0.0] * N
for j, b in enumerate(x_vals):
    somme_rec = 0
    for i, a in enumerate(scales):
        # On reconstruit en sommant sur les échelles et les positions
        wav = (1/math.sqrt(a)) * CHOIX_ONDELETTE((b - x_vals[j])/a) # Simplifié
        # Formule simplifiée de l'intégrale inverse
        somme_rec += coeffs[i, j] * (1 / a**2) 
    f_rec[j] = somme_rec

# Normalisation finale pour l'affichage (mise à l'échelle)
max_orig = max(abs(f(v)) for v in x_vals)
max_rec = max(abs(v) for v in f_rec) if max(abs(v) for v in f_rec) != 0 else 1
f_rec = [(v / max_rec) * max_orig for v in f_rec]

# --- 5. AFFICHAGE MATPLOTLIB ---

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# 1. Original
y_orig = [f(v) for v in x_vals]
ax1.plot(x_vals, y_orig, color='black', label='Original f(x)')
ax1.set_title(f"Signal Original")
ax1.legend()

# 2. Scalogramme
im = ax2.imshow(coeffs, aspect='auto', extent=[0, 1, max(scales), min(scales)], cmap='jet')
ax2.set_title(f"Scalogramme (Ondelette: {CHOIX_ONDELETTE.__name__})")
ax2.set_ylabel("Échelles (a)")
fig.colorbar(im, ax=ax2)

# 3. Reconstruction
ax3.plot(x_vals, f_rec, color='red', linestyle='--', label='Reconstruit')
ax3.set_title("Reconstruction (Bijection par intégrales)")
ax3.legend()

plt.tight_layout()
plt.show()