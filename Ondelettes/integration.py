import math
from fractions import Fraction
from typing import Callable
import numpy as np


class Function:
    def __init__(self, f: Callable[[float], float]):
        self.f = f

    def eval_int(self):
        raise NotImplementedError


class Integral(Function):
    def __init__(self, f: Callable[[float], float], A: float, B: float, N: int, n: int):
        """
        f : fonction à intégrer
        A : borne inférieure
        B : borne supérieure
        N : nombre de sous-intervalles (pas d'intégration)
        n : degré du polynôme interpolateur (ex: 1=trapèzes, 2=Simpson, 4=Boole)
        """
        super().__init__(f)
        self.A = A
        self.B = B
        self.N = N
        self.n = n

        self._weights = self._compute_weights()

    @staticmethod
    def _elementary_symmetric(roots: list[int]) -> list[Fraction]:
        """
        Calcule les fonctions symétriques élémentaires e_0, e_1, ..., e_n
        de la liste `roots` (entiers), via la récurrence :
            prod(t - r) = sum_k (-1)^k * e_k * t^(n-k)

        Retourne les coefficients du polynôme prod(t - r) dans la base
        [t^n, t^(n-1), ..., t^0] sous forme de Fraction.
        """
        # On construit prod(t - r) par multiplications successives.
        # poly[k] = coefficient de t^k, représenté comme Fraction.
        poly = [Fraction(0)] * (len(roots) + 1)
        poly[0] = Fraction(1)  # polynôme constant 1
        for r in roots:
            # Multiplication par (t - r) : new[k] = old[k-1] - r*old[k]
            new_poly = [Fraction(0)] * (len(poly))
            for k in range(len(poly)):
                if k > 0:
                    new_poly[k] += poly[k - 1]
                new_poly[k] -= Fraction(r) * poly[k]
            poly = new_poly
        return poly  # poly[k] = coeff de t^k

    @staticmethod
    def _integrate_polynomial(coeffs: list[Fraction], a: int, b: int) -> Fraction:
        """
        Intègre analytiquement le polynôme sum_k coeffs[k] * t^k de a à b
        (a, b entiers). 
        Retourne la valeur exacte sous forme de Fraction.
        """
        result = Fraction(0)
        for k, c in enumerate(coeffs):
            # Primitive : c * t^(k+1) / (k+1)
            result += c * (Fraction(b) ** (k + 1) - Fraction(a) ** (k + 1)) / (k + 1)
        return result

    def _integrate_product(self, n: int, i: int) -> Fraction:
        """
        Calcule exactement l'intégrale de 0 à n de prod_{j=0, j!=i}^{n} (t - j) dt.

        Les racines sont directement S = {0,...,n}-{i}, entiers.
        On développe le produit via les fonctions symétriques élémentaires de S,
        puis on intègre chaque monôme analytiquement — tout reste exact dans Q.
        """
        roots = [j for j in range(n + 1) if j != i]   # racines entières : {0,...,n} \\{i}
        coeffs = self._elementary_symmetric(roots)     # coefficients de prod_{j in S}(t - j)
        return self._integrate_polynomial(coeffs, 0, n)

    def _compute_weights(self) -> list[float]:
        """
        Calcule les coefficients C_i (sans Delta) tels que w_i = C_i * Delta.

            w_i = (-1)^(n-i) * C(n,i) / n! * I_i

        Tout le calcul est exact, converti en float à la fin.
        """
        n = self.n
        fact_n = Fraction(math.factorial(n))
        coeffs = []
        for i in range(n + 1):
            I_i   = self._integrate_product(n, i)
            binom = Fraction(math.comb(n, i))
            sign  = Fraction((-1) ** (n - i))
            C_i   = sign * binom * I_i / fact_n
            coeffs.append(float(C_i))
        return coeffs

    def eval_int(self) -> float:
        """
        Calcule l'intégrale de A à B de f par la méthode de Newton-Cotes composée.
        Complexité : O(N * n) évaluations de f.
        """
        A, B, N, n = self.A, self.B, self.N, self.n
        h = (B - A) / N          # largeur d'un sous-intervalle
        Delta = h / n            # pas entre les points d'interpolation

        total = 0.0
        for k in range(N):
            a_k = A + k * h
            for i in range(n + 1):
                x_i = a_k + i * Delta
                total += self.f(x_i) * self._weights[i]

        return total * Delta

class Fourier(Function):
    """
    Calcule la transformée de Fourier d'une fonction f (intégrable au sens de Lebesgue)
    par intégration numérique de Newton-Cotes.

    Définition retenue (convention physique) :
        F(ν) = ∫_{T_min}^{T_max} f(t) · exp(-2πi ν t) dt

    On décompose en parties réelle et imaginaire, chacune intégrée séparément
    avec la classe Integral (Boole n=2 par défaut).
    """

    def __init__(self, f: Callable[[float], float],
                 T_min: float, T_max: float,
                 N: int = 1000, n: int = 4):
        """
        f     : signal temporel (fonction réelle ou complexe)
        T_min : début de la fenêtre d'intégration
        T_max : fin   de la fenêtre d'intégration
        N     : nombre de sous-intervalles pour Newton-Cotes
        n     : degré du polynôme interpolateur
        """
        super().__init__(f)
        self.T_min = T_min
        self.T_max = T_max
        self.N = N
        self.n = n

    def at(self, nu: float) -> complex:
        """
        Retourne F(ν) = ∫ f(t) exp(-2πi ν t) dt  pour une fréquence ν donnée.
        """
        f_re = lambda t: self.f(t) * math.cos(2 * math.pi * nu * t)
        f_im = lambda t: -self.f(t) * math.sin(2 * math.pi * nu * t)

        re = Integral(f_re, self.T_min, self.T_max, self.N, self.n).eval_int()
        im = Integral(f_im, self.T_min, self.T_max, self.N, self.n).eval_int()
        return re +1j*im

    def spectrum(self, nu_min: float, nu_max: float,
                 n_freqs: int = 200) -> tuple[list[float], list[complex]]:
        """
        Calcule F(ν) sur une grille de n_freqs fréquences entre nu_min et nu_max.
        Retourne (freqs, values).
        """
        freqs  = [nu_min + k * (nu_max - nu_min) / (n_freqs - 1) for k in range(n_freqs)]
        values = [self.at(nu) for nu in freqs]
        return freqs, values
    
    def module(self, nu_min: float, nu_max: float,
                 n_freqs: int = 200) -> tuple[list[float], list[float]]:
        return map(np.abs(), self.spectrum(nu_min, nu_max, n_freqs))
    
    def argument(self, nu_min: float, nu_max: float,
                 n_freqs: int = 200) -> tuple[list[float], list[float]]:
        return map(np.angle(), self.spectrum(nu_min, nu_max, n_freqs))