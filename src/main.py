#!/usr/bin/env python3

import argparse
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import statistics

import display
import filter


def save_img(img_array, original_path, effect_name):
    """Salva a imagem com o efeito aplicado no mesmo diretório e com o mesmo nome, mas com '_<effect_name>' antes da extensão"""
    new_img = Image.fromarray(img_array)
    path = Path(original_path)
    # save image on the same path and name but with '_<effect_name>' before the extension
    new_img.save(f'{path.parent}/{path.stem}_{effect_name}{path.suffix}')


# TODO: short options
parser = argparse.ArgumentParser()
parser.add_argument('FILE', help='path to input image')
parser.add_argument('--output', choices=['terminal', 'terminal-numbers', 'matplotlib', 'save'], default='matplotlib', help='sets image output method')
parser.add_argument('--grayscale', action='store_true', help='convert image to grayscale')
parser.add_argument('--no-original', action='store_true', help='do not show the original image')
# 1
parser.add_argument('--yiq', action='store_true', help='performs RGB-YIQ-RGB conversion')
# 2
parser.add_argument('--neg-rgb', action='store_true', help='performs RGB negative')
parser.add_argument('--neg-y', action='store_true', help='performs Y negative')
# 3 & 4
# filtros e sequencias de filtros podem ser especificadas multiplas vezes para ver o efeito de cada um
# os filtros especiais (como mediana) devem ser colocados entre colchetes sem espaços (ex: [mediana,3,3,1,1,true]
# o formato desses filtros especiais é: [nome, linhas, colunas, pivô x, pivô y, extensão com zeros]
parser.add_argument('--filter', action='append', help='apply filter to image. Can be used multiple times for multiple filters. Function filters (like median) should be put inside square brackets without spaces (e.g. [median,3,3,1,1,true]')
parser.add_argument('--filter-sequence', nargs='+', action='append', help='apply a sequence of filters to image. Can be used multiple times for multiple filter sequences')

args = parser.parse_args()

IMAGE_PATH = args.FILE
display_handler = {'terminal': display.terminal,
                   'terminal-numbers': display.terminal_numbers,
                   'matplotlib': display.matplotlib_imshow,
                   'save': lambda *args: None # do nothing
                   }[args.output]
save_handler = save_img if args.output == 'save' else lambda *args: None # do nothing

with Image.open(IMAGE_PATH) as img:
    if args.grayscale:
        img = img.convert('L')
    img_arr = np.asarray(img.convert('RGB'))
    img_arr_norm = img_arr / 255


RGB_TO_YIQ_MATRIX = np.array([
                              [0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.322],
                              [0.211, -0.523, 0.312]
                            ])

YIQ_TO_RGB_MATRIX = np.array([
                                [1.000, 0.956, 0.621],
                                [1.000, -0.272, -0.647],
                                [1.000, -1.106, 1.703]
                            ])

if not args.no_original:
    display_handler(img_arr, 'Original Image')

# Realiza uma operação de conversão de espaço de cores da imagem de RGB para YIQ e novamente de volta para RGB
if args.yiq:
    #? TODO: podemos normalizar os valores?
    img_arr_norm_yiq = np.dot(img_arr_norm.copy(), RGB_TO_YIQ_MATRIX.T) # Cria uma cópia do array de imagem normalizado e multiplica o array de imagem  matriz de conversão de RGB para YIQ
    img_arr_norm_rgb = np.dot(img_arr_norm_yiq, YIQ_TO_RGB_MATRIX.T) # Retorna para RGB
    img_arr_neg_rgb = np.clip(img_arr_norm_rgb * 255, 0, 255).round().astype(np.uint8) # Multiplica o array por 255 e limita seus valores entre 0 e 255 usando a função np.clip()
    display_handler(img_arr_neg_rgb, 'RGB-YIQ-RGB Image') # exibir
    save_handler(img_arr_neg_rgb, IMAGE_PATH, 'yiq')

# Negativo em RGB
if args.neg_rgb:
    img_arr_neg_rgb = (255 - img_arr).astype(np.uint8) 
    display_handler(img_arr_neg_rgb, 'RGB Negative Image')
    save_handler(img_arr_neg_rgb, IMAGE_PATH, 'neg_rgb')

# Negativo em Y
if args.neg_y:
    img_arr_norm_yiq = np.dot(img_arr_norm.copy(), RGB_TO_YIQ_MATRIX.T)
    img_arr_norm_yiq[:, :, 0] = 1 - img_arr_norm_yiq[:, :, 0] # Inverte Y
    img_arr_norm_rgb = np.dot(img_arr_norm_yiq, YIQ_TO_RGB_MATRIX.T)
    img_arr_rgb_neg_y = np.clip(img_arr_norm_rgb * 255, 0, 255).round().astype(np.uint8)
    display_handler(img_arr_rgb_neg_y, 'Y Negative Image')
    save_handler(img_arr_rgb_neg_y, IMAGE_PATH, 'neg_y')

# Verifica se a opção filter foi passada como argumento
if args.filter:
    # Itera sobre cada filtro passado como argumento
    for f in args.filter:
        if f[0] == '[' and f[-1] == ']': # Se o elemento for uma string delimitada por colchetes [ e ], ele é tratado como uma função definida pelo usuário
            new_filter = filter.get_function_filter(f)
        else: # Caso contrário, considera que é um filtro em formato JSON e utiliza o método from_json do módulo filter para obter a instância do filtro
            new_filter = filter.DataFilter.from_json(f)
        # Aplica o filtro na imagem, criando uma cópia da matriz de pixels
        img_arr_filtered = new_filter.apply(img_arr.copy())
        display_handler(img_arr_filtered, f'Filtered Image with {new_filter.name}')
        save_handler(img_arr_filtered, IMAGE_PATH, f'filter-{new_filter.name}')

# Verifica se há uma sequência de filtros especificada nos argumentos
if args.filter_sequence:
    # Itera sobre a lista de sequências de filtros especificada nos argumentos
    for fs in args.filter_sequence:
        # Faz uma cópia da imagem original para ser filtrada
        img_arr_filtered = img_arr.copy()
        for f in fs:
            # Verifica se o filtro é uma função pré-definida
            if f[0] == '[' and f[-1] == ']':
                # Obtém a função de filtro a partir da string especificada
                new_filter = filter.get_function_filter(f)
                
            # Se não for uma função pré-definida, verifica se o formato de entrada dos filtros é JSON
            else:
                # Cria um objeto de filtro de dados a partir da string JSON especificada
                if args.input_format == 'json':
                    new_filter = filter.DataFilter.from_json(f)
                # Se o formato de entrada dos filtros for TXT, cria um objeto de filtro de dados a partir da string TXT especificada
                else:
                    new_filter = filter.DataFilter.from_txt(f)
            img_arr_filtered = new_filter.apply(img_arr_filtered)
        display_handler(img_arr_filtered, f'Filtered Image with {fs}')
        save_handler(img_arr_filtered, IMAGE_PATH, f'filter sequence-{fs}')

