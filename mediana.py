from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def imagemArray(imagem):  # Transforma a imagem em array
    return np.asarray(imagem)

def exibe(imagem_resultante,title): #Definida uma função para exibição das imagens resultantes
  plt.title(title) #Coloca título na imagem
  plt.imshow(imagem_resultante) #Usado comando plt.imshow para transformar matriz ou array em imagem
  plt.show() #Plota a imagem

#m_med, n_med, mask_media
def lermascara():
    with open("sobel.txt", "r") as f:
        # Ler as dimensões da máscara
        rows = int(f.readline().strip())
        cols = int(f.readline().strip())

        # Ler os valores da máscara
        mascara = []
        for i in range(rows):
            linha = [float(x) for x in f.readline().strip().split()]
            mascara.append(linha)

        #print(rows, cols)
        #print(mascara)
        mascara_np = np.array(mascara)
    return mascara_np

def correlacao(img, dimensaofiltromediana):
    imagemresultante = np.zeros_like(img)
    # Obter a altura, largura e o RGB (var não utilizada) da imagem
    imageheight, imagewidth, _ = img.shape
    # Obter a altura e largura da máscara
    maskheight, maskwidth = dimensaofiltromediana, dimensaofiltromediana

   

    for h in range(imageheight):
        # Definir o início da coluna e o fim da coluna da submatriz
        coluna_inicio = h
        coluna_fim = coluna_inicio + maskheight

        # Verificar se o índice é inválido e ajustá-lo
        if coluna_fim >= imageheight:
            coluna_fim = imageheight
            coluna_inicio = coluna_fim - maskheight

        if coluna_inicio < 0:
            coluna_inicio = 0
            coluna_fim = coluna_inicio + maskheight

        # Definir o início da linha e o fim da linha da submatriz
        for w in range(imagewidth):
            linha_inicio = w
            linha_fim = linha_inicio + maskwidth

            # Check invalid indexes
            if linha_fim >= imagewidth:
                linha_fim = imagewidth
                linha_inicio = linha_fim - maskwidth

            if linha_inicio < 0:
                linha_inicio = 0
                linha_fim = linha_inicio + maskwidth

            # Obter a submatriz para cada canal de cor
            for c in range(img.shape[2]):
                submatrix = img[coluna_inicio:coluna_fim, linha_inicio:linha_fim, c]
                # get the median value of the submatrix
                imagemresultante[h, w, c] = np.median(submatrix)

            

    return imagemresultante

dimensaofiltromediana = 5

imagemOriginal = Image.open('ci.png')
arrayOriginal = imagemArray(imagemOriginal)
larguraOriginal, alturaOriginal = imagemOriginal.size

imagemResultante = correlacao(arrayOriginal, dimensaofiltromediana)
imagemResultante = imagemArray(imagemResultante)
exibe(imagemResultante, "Imagem Resultante")