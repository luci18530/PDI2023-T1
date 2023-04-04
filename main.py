#!/usr/bin/env python3

import argparse
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='path to input image')
args = parser.parse_args()
IMAGE_PATH = args.image_path

def show_img_terminal(img_arr):
    str = ''
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            #print(f'\x1b[48;2;{img_arr[i][j][0]};{img_arr[i][j][1]};{img_arr[i][j][2]}m  ', end='')
            str += f'\x1b[48;2;{img_arr[i][j][0]};{img_arr[i][j][1]};{img_arr[i][j][2]}m  '
        str += '\x1b[0m\n'
    print(str)
    print('\x1b[0m', end='')


def show_img_terminal_numbers(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            print(img_arr[i][j], end='')
        print('')


img = Image.open(IMAGE_PATH)
img_arr = np.asarray(img)
print('Original image:')
show_img_terminal(img_arr)

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

img_arr_yiq = np.dot(img_arr / 255, RGB_TO_YIQ_MATRIX.T)
#show_img_terminal_numbers(img_arr_yiq)
print('RGB-YIQ-RGB image:')
show_img_terminal((np.dot(img_arr_yiq, YIQ_TO_RGB_MATRIX.T) * 255).round().astype(int))

