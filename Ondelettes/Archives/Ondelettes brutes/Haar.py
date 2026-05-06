from random import randint
import numpy as np

def Haar(a,b):
    return -a+b

def moyenne(a,b):
    return (a+b)/2

def est_puissance_2(L,N):
    for i in range(N+1):
        if len(L) == 2**i:
            return True
    return False

def ondelettes(L):
    if not est_puissance_2(L,N):
        return None
    if len(L) < 2:
        return L
    res,moy = [],[]
    for i in range(0,len(L)-1,2):
        res.append(Haar(L[i],L[i+1]))
        moy.append(moyenne(L[i],L[i+1]))
    return res + ondelettes(moy)

def decompression(L):
    if not est_puissance_2(L,N):
        return
    n = len(L)
    copie = L[:]
    for i in range(int(np.log2(len(L)))):
        T = []
        moy = copie[(n-2**i):]
        res = copie[:(n-2**i)]
        for j in range(len(moy)):
            a = res.pop(-1)/2
            T = T + [moy[j]+a] + [moy[j]-a]
        copie = res + T
    return copie[::-1]

def fl2int(L):
    for i in range(len(L)):
        if isinstance(L[i],float):
            L[i] = int(L[i])
    return L

def sont_egales(L1,L2):
    if len(L1) != len(L2):
        return False
    for i in range(len(L1)):
        if L1[i] != L2[i]:
            return False
    return True

N = 10
TAILLE = 7
L = [randint(0,100) for _ in range(2**TAILLE)]
L_comp = ondelettes(L)
L_decomp = fl2int(decompression(L_comp))
print(sont_egales(L,L_decomp))