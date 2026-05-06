def haar(a,b):
    return -a+b

def moyenne(a,b):
    return (a+b)/2

def compression_ondelettes(L,r=3):
    if len(L)==1:
        return [L]
    res, moy = [], []
    for i in range(0, len(L)-1, 2):
        a = L[i]
        b = L[i+1]
        moy.append(moyenne(a,b))
        res.append(round(haar(a,b), r))
    return [res] + compression_ondelettes(moy)

def decompression_ondelettes(C):
    L = C[-1]
    for i in range(len(C) - 2, -1, -1):
        coefficients = C[i]
        nouvelle_L = []
        
        for j in range(len(L)):
            m = L[j]            # La moyenne actuelle
            h = coefficients[j] # Le détail (Haar)
    
            a = m - h/2
            b = m + h/2
            
            nouvelle_L.append(a)
            nouvelle_L.append(b)
            
        L = nouvelle_L
        
    return L

#####

def compression_ondelettes(L,r=3):
    if len(L)==1:
        return [L]
    res, moy = [], []
    for i in range(0, len(L)-1, 2):
        a = L[i]
        b = L[i+1]
        moy.append(moyenne(a,b))
        res.append(round(haar(a,b), r))
    return [res] + compression_ondelettes(moy)

L = [4,6,7,8,8,8,9,7]
print(L)

def decompression_ondelettes_plate(L):
    if len(L) == 1:
        return L

    milieu = len(L) // 2
    details = L[:milieu]
    reste = L[milieu:]

    moyennes = decompression_ondelettes_plate(reste)

    res = []
    for i in range(len(moyennes)):
        m = moyennes[i]
        h = details[i]
        
        # Formules inverses :
        # a = m - h/2
        # b = m + h/2
        res.append(m - h/2)
        res.append(m + h/2)

    return res