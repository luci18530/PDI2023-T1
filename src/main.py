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
    new_img = Image.fromarray(img_array)
    path = Path(original_path)
    # save image on the same path and name but with '_<effect_name>' before the extension
    new_img.save(f'{path.parent}/{path.stem}_{effect_name}{path.suffix}')


# TODO: short options
# TODO: add pipeline option instead of filter sequence
parser = argparse.ArgumentParser()
parser.add_argument('FILE', help='path to input image. defaults to stdin')
parser.add_argument('--output', choices=['terminal', 'terminal-numbers', 'matplotlib', 'save'], default='terminal', help='sets output method')
parser.add_argument('--input-format', choices=['txt', 'json'], default='txt', help='sets input format')
parser.add_argument('--no-original', action='store_true', help='do not show original image')
# 1
parser.add_argument('--yiq', action='store_true', help='performs RGB-YIQ-RGB conversion')
# 2
parser.add_argument('--neg-rgb', action='store_true', help='performs RGB negative')
parser.add_argument('--neg-y', action='store_true', help='performs Y negative')
# 3 & 4
parser.add_argument('--filter', action='append', help='apply filter to image. Can be used multiple times for multiple filters. Function filters (like median) should be put inside square brackets without spaces (e.g. [median,3,3,1,1,true]')
parser.add_argument('--filter-sequence', nargs='+', action='append', help='apply a sequence of filters to image. Can be used multiple times for multiple filter sequences')
# 5
parser.add_argument('--correlation', nargs='+', action='append', help='apply correlation to image. Can be used multiple times for multiple correlations. <n> says how many correlations to perform in a single image')

args = parser.parse_args()

# TODO: filters on files shouldn't be 3D?

IMAGE_PATH = args.FILE
display_handler = {'terminal': display.terminal,
                   'terminal-numbers': display.terminal_numbers,
                   'matplotlib': display.matplotlib_imshow,
                   'save': lambda *args: None # do nothing
                   }[args.output]
save_handler = save_img if args.output == 'save' else lambda *args: None # do nothing

with Image.open(IMAGE_PATH) as img:
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

if args.yiq:
    #? TODO: podemos normalizar os valores?
    img_arr_norm_yiq = np.dot(img_arr_norm.copy(), RGB_TO_YIQ_MATRIX.T)
    img_arr_norm_rgb = np.dot(img_arr_norm_yiq, YIQ_TO_RGB_MATRIX.T)
    img_arr_neg_rgb = np.clip(img_arr_norm_rgb * 255, 0, 255).round().astype(np.uint8)
    display_handler(img_arr_neg_rgb, 'RGB-YIQ-RGB Image')
    save_handler(img_arr_neg_rgb, IMAGE_PATH, 'yiq')

if args.neg_rgb:
    img_arr_neg_rgb = (255 - img_arr).astype(np.uint8)
    display_handler(img_arr_neg_rgb, 'RGB Negative Image')
    save_handler(img_arr_neg_rgb, IMAGE_PATH, 'neg_rgb')

if args.neg_y:
    img_arr_norm_yiq = np.dot(img_arr_norm.copy(), RGB_TO_YIQ_MATRIX.T)
    img_arr_norm_yiq[:, :, 0] = 1 - img_arr_norm_yiq[:, :, 0]
    img_arr_norm_rgb = np.dot(img_arr_norm_yiq, YIQ_TO_RGB_MATRIX.T)
    img_arr_rgb_neg_y = np.clip(img_arr_norm_rgb * 255, 0, 255).round().astype(np.uint8)
    display_handler(img_arr_rgb_neg_y, 'Y Negative Image')
    save_handler(img_arr_rgb_neg_y, IMAGE_PATH, 'neg_y')

if args.filter:
    for f in args.filter:
        if f[0] == '[' and f[-1] == ']':
            new_filter = filter.get_function_filter(f)
        else:
            if args.input_format == 'json':
                new_filter = filter.DataFilter.from_json(f)
            else:
                new_filter = filter.DataFilter.from_txt(f)
        img_arr_filtered = new_filter.apply(img_arr.copy())
        display_handler(img_arr_filtered, f'Filtered Image with {new_filter.name}')
        save_handler(img_arr_filtered, IMAGE_PATH, f'filter-{new_filter.name}')

if args.filter_sequence:
    for fs in args.filter_sequence:
        img_arr_filtered = img_arr.copy()
        for f in fs:
            if f[0] == '[' and f[-1] == ']':
                new_filter = filter.get_function_filter(f)
            else:
                if args.input_format == 'json':
                    new_filter = filter.DataFilter.from_json(f)
                else:
                    new_filter = filter.DataFilter.from_txt(f)
            img_arr_filtered = new_filter.apply(img_arr_filtered)
        display_handler(img_arr_filtered, f'Filtered Image with {fs}')
        save_handler(img_arr_filtered, IMAGE_PATH, f'filter sequence-{fs}')

