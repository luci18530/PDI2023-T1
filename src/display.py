# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

# função que imprime a imagem no terminal
def _terminal_img_printing(img_array, pixel_function, end_line, end):
    # inicializa a string que armazenará os caracteres de cada pixel
    str = ''
    # itera sobre cada pixel da imagem
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # chama a função pixel_function para obter a representação do pixel
            str += pixel_function(img_array[i][j])
        # adiciona uma quebra de linha ao final de cada linha da imagem
        str += '\x1b[0m\n'
    print(str) # imprime a imagem
    print('\x1b[0m', end='') # imprime o caractere de reset de cor ao final da imagem


# função que exibe a imagem no terminal com cores
def terminal(img_array, name='Resulting Image'):
    print(name)
                                            # colors on terminal                  # reset color 
    _terminal_img_printing(img_array, lambda x: f'\x1b[48;2;{x[0]};{x[1]};{x[2]}m  ', '\x1b[0m\n', '\x1b[0m')

# função que exibe a imagem no terminal como uma matriz de números RGB
def terminal_numbers(img_arr, name='Resulting Image'):
    print(name)
    _terminal_img_printing(img_arr, lambda x: f'[{x[0]:03d},{x[1]:03d},{x[2]:03d}]', '', '')

# função que exibe a imagem usando o matplotlib
def matplotlib_imshow(img_array, name='Resulting Image'):
    plt.title(name)
    plt.imshow(img_array)
    plt.show()

# função que salva a imagem em um arquivo .png
def save(img_array, name='Resulting Image'):
    plt.title(name)
    plt.imshow(img_array)
    plt.savefig('result.png')
