from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def imagemArray(imagem):  # Transforma a imagem em array
    return np.asarray(imagem)

def exibe(imagem_resultante,title): #Definida uma função para exibição das imagens resultantes
  plt.title(title) #Coloca título na imagem
  plt.imshow(imagem_resultante) #Usado comando plt.imshow para transformar matriz ou array em imagem
  plt.show() #Plota a imagem

def checar255(x):
    x = np.rint(x)
    if x > 255:
        return 255
    elif x < 0:
        return 0
    return x

def negativeRGB(imagemRGB, largura, altura):
    imagemRGB = imagemRGB.astype(int)
    for h in range(altura):
        for w in range(largura):
            r, g, b = imagemRGB[h][w]
            r = 255 - r
            g = 255 - g
            b = 255 - b
            imagemRGB[h][w] = [r, g, b]
    imagemRGB = np.uint8(imagemRGB)
    return imagemRGB

#  CONVERSÃO RGB - YIQ com Y negativo -------------------------------------------------------------


def RGBYIQ(imagemRGB, largura, altura):  # Converte a imagem de RGB para YIQ

    # Copia a imagem e transforma array em float para os calculos de conversão
    imagemYIQ = imagemRGB.copy()
    imagemYIQ = imagemYIQ.astype(float)

    for h in range(altura):
        for w in range(largura):
            r, g, b = imagemRGB[h][w]
            y = 0.299*r + 0.587*g + 0.114*b
            i = 0.596*r - 0.274*g - 0.322*b
            q = 0.211*r - 0.523*g + 0.312*b
            imagemYIQ[h][w] = [y, i, q]

    # negative of Y channel
    for h in range(altura):
        for w in range(largura):
            y, i, q = imagemYIQ[h][w]
            y = 255 - y
            imagemYIQ[h][w] = [y, i, q]

    return imagemYIQ

# --------------------------------- CONVERSÃO YIQ - RGB --------------------------------------------

def converterYIQRGB(y, i, q): # Converte a imagem de YIQ para RGB
    r = int(1.000*y + 0.956*i + 0.621*q)
    g = int(1.000*y - 0.272*i - 0.647*q)
    b = int(1.000*y - 1.106*i + 1.703*q)
    return checar255(r), checar255(g), checar255(b)

def YIQRGB(imagemYIQ, largura, altura):
    # Convertendo imagem para int (arrendondando)
    imagemRGB = imagemYIQ.astype(int)

    for h in range(altura):
        for w in range(largura):
            y, i, q = imagemYIQ[h][w]
            r, g, b = converterYIQRGB(y, i, q)
            imagemRGB[h][w] = [r, g, b]

    imagemRGB = np.uint8(imagemRGB) # Convertendo imagem para uint8
    return imagemRGB
# --------------------------------- MAIN -----------------------------------------------------------

# --------------------------------- ORIGINAL -------------------------------------------------------
imagemOriginal = Image.open('ci.png')
arrayOriginal = imagemArray(imagemOriginal)
larguraOriginal, alturaOriginal = imagemOriginal.size

# --------------------------------- RGB ------------------------------------------------------------
arrayModificada = RGBYIQ(arrayOriginal, larguraOriginal, alturaOriginal)
imagemModificada = imagemArray(arrayModificada)
exibe(imagemModificada, "Imagem YIQ")

# --------------------------------- YIQ NEGATIVE ---------------------------------------------------
arrayModificada = YIQRGB(arrayModificada, larguraOriginal, alturaOriginal)
imagemModificada = imagemArray(arrayModificada)
exibe(imagemModificada, "Imagem RGB com canal Y negativo")

# --------------------------------- RGB NEGATIVE ---------------------------------------------------
imagemRGBnegativo = negativeRGB(arrayOriginal, larguraOriginal, alturaOriginal)
exibe(imagemRGBnegativo, "Imagem RGB negativo")
