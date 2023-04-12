# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

def _terminal_img_printing(img_array, pixel_function, end_line, end):
    str = ''
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            str += pixel_function(img_array[i][j])
        str += '\x1b[0m\n'
    print(str)
    print('\x1b[0m', end='')


def terminal(img_array, name='Resulting Image'):
    print(name)
                                            # colors on terminal                  # reset color 
    _terminal_img_printing(img_array, lambda x: f'\x1b[48;2;{x[0]};{x[1]};{x[2]}m  ', '\x1b[0m\n', '\x1b[0m')


def terminal_numbers(img_arr, name='Resulting Image'):
    print(name)
    _terminal_img_printing(img_arr, lambda x: f'[{x[0]:03d},{x[1]:03d},{x[2]:03d}]', '', '')


def matplotlib_imshow(img_array, name='Resulting Image'):
    plt.title(name)
    plt.imshow(img_array)
    plt.show()


def save(img_array, name='Resulting Image'):
    plt.title(name)
    plt.imshow(img_array)
    plt.savefig('result.png')
