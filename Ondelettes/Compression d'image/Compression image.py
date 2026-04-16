from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

## Extraction image
def extraction_image(NomFichier):
    img = Image.open(NomFichier)
    return np.array(img)[:,:,:].tolist()

## Compression image
def compression(img):
    n = len(img)
    L = []
    img_comp = []
    pixel_comp = []
    colonne = []
    for i in range(n):
        for j in range(0,n-1,2):
            for k in range(3):
                pixel_comp.append((img[i][j][k]+img[i][j+1][k])//2)
            colonne.append(pixel_comp)
            pixel_comp = []
        L.append(colonne)
        colonne = []
    for i in range(n//2):
        for j in range(0,n-1,2):
            for k in range(3):
                pixel_comp.append((L[j][i][k]+L[j+1][i][k])//2)
            colonne.append(pixel_comp)
            pixel_comp = []
        img_comp.append(colonne)
        colonne = []
    return img_comp

## Upscale image
def upscale_rapport2(L):
    n = len(L)
    L_upscaled = []
    ligne_upscaled = []
    for i in range(n):
        ligne = []
        for j in range(n):
            ligne += [L[i][j]]*2
        ligne_upscaled.append(ligne)
    for i in range(n*2):
        colonne = []
        for j in range(n):
            colonne += [ligne_upscaled[j][i]]*2
        L_upscaled.append(colonne)
    return L_upscaled

## Retournement image
def retourne(img):
    N = len(img)
    for i in range(N):
        for j in range(i):
            img[i][j],img[j][i] = img[j][i],img[i][j]

## Affichage image
def affichage(img):
    plt.imshow(img)
    plt.show()

## Test
img = extraction_image('maison.jpg')
img_comp = compression(img)
retourne(img_comp)
# img_upscaled = upscale_rapport2(img_comp)
# retourne(img_upscaled)
affichage(img_comp)