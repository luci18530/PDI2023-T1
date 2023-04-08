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
def lermascara(arquivo):
    with open(arquivo, "r") as f:
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

def correlacao(img, mascara):
    # Criar matriz de zeros para armazenar o resultado da correlação
    imagemresultante = np.zeros_like(img)
    # Obter a altura, largura e o RGB (var não utilizada) da imagem
    imageheight, imagewidth, _ = img.shape
    # Obter a altura e largura da máscara
    maskheight, maskwidth = mascara.shape

    for w in range(imagewidth):
        # Definir o início da linha e o fim da linha da submatriz
        linha_inicio = w
        linha_fim = linha_inicio + maskwidth

        # Verificar se o índice é inválido e ajustá-lo
        if linha_fim >= imagewidth:
            linha_fim = imagewidth
            linha_inicio = linha_fim - maskwidth

        # Definir o início da coluna e o fim da coluna da submatriz
        for h in range(imageheight):
            coluna_inicio = h
            coluna_fim = coluna_inicio + maskheight

            # Verificar se o índice é inválido e ajustá-lo
            if coluna_fim >= imageheight:
                coluna_fim = imageheight
                coluna_inicio = coluna_fim - maskheight

            # Obter a submatriz para cada canal de cor
            for c in range(img.shape[2]):
                submatrix = img[coluna_inicio:coluna_fim, linha_inicio:linha_fim, c]
                submatrix = submatrix * mascara
                # if the sum get negative set it to 0
                if np.sum(submatrix) < 0:
                    submatrix = 0

                # Somar os valores
                imagemresultante[h, w, c] = (np.sum(submatrix).astype(np.uint8))
                

    imagemresultante = np.abs(imagemresultante)
    imagemresultante = (imagemresultante / np.max(imagemresultante)) * 255
    return imagemresultante.astype(np.uint8)

mascara_sobel_horizontal = lermascara("sobel_horizontal.txt")
mascara_sobel_vertical = lermascara("sobel_vertical.txt")

imagemOriginal = Image.open('ci.png')
arrayOriginal = imagemArray(imagemOriginal)
larguraOriginal, alturaOriginal = imagemOriginal.size

imagemResultante = correlacao(arrayOriginal, mascara_sobel_horizontal)
imagemResultante = imagemArray(imagemResultante)
exibe(imagemResultante, "Imagem com filtro de Sobel Horizontal")

imagemResultante2 = correlacao(arrayOriginal, mascara_sobel_vertical)
imagemResultante2 = imagemArray(imagemResultante2)
exibe(imagemResultante2, "Imagem com filtro de Sobel Vertical")

# merge the two sobel images
imagemResultante3 = cv2.addWeighted(imagemResultante, 0.5, imagemResultante2, 0.5, 0)
exibe(imagemResultante3, "Imagem com filtro de Sobel")
