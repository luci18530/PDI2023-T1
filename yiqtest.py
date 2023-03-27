from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def imagemArray(imagem):  # Transforma a imagem em array
    return np.asarray(imagem)

def exibe(imagem_resultante): #Definida uma função para exibição das imagens resultantes
  plt.title("Imagem Resultante") #Coloca título na imagem
  plt.imshow(imagem_resultante) #Usado comando plt.imshow para transformar matriz ou array em imagem
  plt.show() #Plota a imagem

def checar255(x):
    x = np.rint(x)
    if x > 255:
        return 255
    elif x < 0:
        return 0
    return x

#  CONVERSÃO RGB - YIQ -----------------------------------------------------------


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

    return imagemYIQ

# --------------------------------- CONVERSÃO YIQ - RGB -----------------------------------------------------------

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

imagemOriginal = Image.open('ci.png')
arrayOriginal = imagemArray(imagemOriginal)
larguraOriginal, alturaOriginal = imagemOriginal.size

arrayModificada = RGBYIQ(arrayOriginal, larguraOriginal, alturaOriginal)
imagemModificada = imagemArray(arrayModificada)
cv2.imshow("imagem_resultante.jpg", imagemModificada)
cv2.imwrite("imagem_resultante.jpg", imagemModificada)
cv2.waitKey(0)
exibe(imagemModificada)

arrayModificada = YIQRGB(arrayModificada, larguraOriginal, alturaOriginal)
imagemModificada = imagemArray(arrayModificada)

exibe(imagemModificada)

imagemModificada = cv2.cvtColor(imagemModificada, cv2.COLOR_BGR2RGB)
cv2.imshow("imagem_resultante.jpg", imagemModificada)
cv2.waitKey(0)
cv2.imwrite("imagem_resultante2.jpg", imagemModificada)