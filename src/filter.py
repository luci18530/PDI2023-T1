# -*- coding: utf-8 -*-

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from functools import partial
import statistics as stats

# funções que limitam valores de pixels entre 0 e 255
# clip: limita valores entre 0 e 255, valores abaixo de 0 são 0 e valores acima de 255 são 255
# absolute: aplicar valor absoluto a cada pixel, limitando entre 0 e 255
# renormalise: normaliza os valores de cada pixel para o intervalo [0, 255]
LIMIT_FUNCTIONS = {
	'clip': lambda arr: np.clip(arr, 0, 255),
	#? TODO: clip or renormalise?
	'absolute': lambda arr:np.clip(np.array([np.abs(band) for band in arr]), 0, 255),
	'renormalise': lambda arr: np.array([((band - np.min(band)) / (np.max(band) - np.min(band))) * 255 for band in arr]),
}

@dataclass(frozen=True)
class AbstractFilter:
	"""classe base para filtros"""
	name: str
	kernel: np.array
	pivot: tuple
	zero_extension: bool
	limit_function: callable
	offset: int

	def __post_init__(self):
		self.self_check()

	def self_check(self):
		"""checa se o filtro é válido"""
		if any(len(row) != len(self.kernel[0]) for row in self.kernel):
			raise ValueError(f'Invalid kernel for filter {self.name}. Not all rows have the same length')

		# if the kernel has only one channel, repeat it 3 times to make it 3 channels
		if len(self.kernel.shape) == 2:
			# self.array = np.repeat(self.array[:, :, np.newaxis], 3, axis=2)
			object.__setattr__(self, "kernel", np.repeat(self.kernel[:, :, np.newaxis], 3, axis=2))

		if all(dim < 1 for dim in self.kernel.shape):
			raise ValueError(f'Invalid kernel for filter {self.name}. Filter must have at least one row and one column')

		if any(len(pixel) != 3 for row in self.kernel for pixel in row):
			raise ValueError(f'Invalid kernel for filter {self.name}. Filter "pixels" must have 1 or 3 channels')

		if len(self.pivot) != 2:
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot must have 2 dimensions')

		# check if pivot elements are integers
		if not all(isinstance(element, int) for element in self.pivot):
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot elements must be integers')

		if self.pivot[0] >= len(self.kernel) or self.pivot[1] >= len(self.kernel[0]):
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot is out of bounds')

		if not isinstance(self.zero_extension, bool):
			raise ValueError(f'Invalid zero_extension for filter {self.name}. zero_extension must be a boolean')

		# check if offset is an integer
		if not isinstance(self.offset, int):
			raise ValueError(f'Invalid offset for filter {self.name}. Offset must be an integer')

	def apply(self, image_array):
		"""aplica o filtro em uma imagem"""
		# calcula quanto padding é necessário para aplicar o filtro
		padding = {
			'row': {
				'before': self.pivot[0],
				'after': self.kernel.shape[0] - self.pivot[0] - 1
			},
			'column': {
				'before': self.pivot[1],
				'after': self.kernel.shape[1] - self.pivot[1] - 1
			}
		}
		if self.zero_extension:
			img_padded = np.pad(image_array,
								((padding['row']['before'], padding['row']['after']),
								 (padding['column']['before'], padding['column']['after']),
								 (0, 0)),
								'constant', constant_values=0)
			# area de aplicação de filtro (por onde o pivô vai passar)
			apply_area = {
				'row': {
					'from': padding['row']['before'],
					'to': padding['row']['before'] + image_array.shape[0]
				},
				'column': {
					'from': padding['column']['before'],
					'to': padding['column']['before'] + image_array.shape[1]
				}
			}
		else:
			img_padded = image_array.copy()
			# area de aplicação de filtro (por onde o pivô vai passar)
			# remove as bordas em que o pivô não consegue passar da área de aplicação
			apply_area = {
				'row': {
					'from': padding['row']['before'],
					'to': image_array.shape[0] - padding['row']['after'] 
				},
				'column': {
					'from': padding['column']['before'],
					'to': image_array.shape[1] - padding['column']['before']
				}
			}

		# nova imagem com as dimensões da área de aplicação
		new_img = np.empty((apply_area['row']['to'] - apply_area['row']['from'], apply_area['column']['to'] - apply_area['column']['from'], 3))
		# aplica o filtro para cada pixel da área de aplicação
		for i, row in enumerate(range(apply_area['row']['from'], apply_area['row']['to'])):
			for j, column in enumerate(range(apply_area['column']['from'], apply_area['column']['to'])):
				# chama _filter_op para aplicar o filtro
				new_img[i, j, :] = self.limit_function(self._filter_op(img_padded[row - padding['row']['before']:row + padding['row']['after'] + 1, column - padding['column']['before']:column + padding['column']['after'] + 1]) + self.offset)

		return new_img.round().astype(np.uint8)
	
	def _filter_op(self, apply_area_array):
		"""aplica o filtro em uma área de aplicação"""
		# não implementado na classe base
		raise NotImplementedError(f'Trying to apply abstract filter. Use one of the subclasses instead. Filter name: {self.name}')


@dataclass(frozen=True)
class DataFilter(AbstractFilter):
	"""classe para filtros definidos em um arquivo json"""
	def from_json(path):
		path = Path(path)
		with path.open() as file:
			filter_json = json.load(file)
			return DataFilter(path.stem,
							  np.array(filter_json['kernel']),
							  tuple(filter_json['pivot']),
							  filter_json['zero_extension'],
							  LIMIT_FUNCTIONS[filter_json['limit_function']],
							  filter_json['offset'])

	def _filter_op(self, apply_area_array):
		return np.sum(apply_area_array * self.kernel, axis=(0, 1))


@dataclass(frozen=True, init=False)
class FunctionFilter(AbstractFilter):
	"""classe para filtros definidos por uma função"""
	func: callable

	def __init__(self, name, rows, columns, pivot, zero_extension, func):
		super().__init__(name, np.empty((rows, columns, 3)), pivot, zero_extension, LIMIT_FUNCTIONS['clip'], offset=0)
		object.__setattr__(self, "func", func)

	def _filter_op(self, apply_area_array):
		return self.func(apply_area_array)


BOX_FILTER = partial(FunctionFilter, '[box]', func=lambda array: [np.mean(array[:,:,band]) for band in range(array.shape[2])])
MEDIAN_FILTER = partial(FunctionFilter, '[median]', func=lambda array: [np.median(array[:,:,band]) for band in range(array.shape[2])])
MODE_FILTER = partial(FunctionFilter, '[mode]', func=lambda array: [stats.mode(array[:,:,band].flatten()) for band in range(array.shape[2])])
ERODE_FILTER = partial(FunctionFilter, '[erode]', func=lambda array: [np.min(array[:,:,band]) for band in range(array.shape[2])])
DILATE_FILTER = partial(FunctionFilter, '[dilate]', func=lambda array: [np.max(array[:,:,band]) for band in range(array.shape[2])])

FUNCTIONS_FILTER_TABLE = {
	'box': BOX_FILTER,
	'mean': BOX_FILTER,
	'median': MEDIAN_FILTER,
	'mode': MODE_FILTER,
	'erode': ERODE_FILTER,
	'min': ERODE_FILTER,
	'dilate': DILATE_FILTER,
	'max': DILATE_FILTER
}

def get_function_filter(filter_str):
	"""retorna um filtro definido por uma função
	faz o parsing da string e retorna o filtro correspondente"""
	# remove brackets
	filter_str = filter_str[1:-1]
	name, rows, columns, pivot_x, pivot_y, zero_extension = filter_str.split(',')
	if name not in FUNCTIONS_FILTER_TABLE:
		raise ValueError(f'Invalid function filter name [{name}]')

	rows = int(rows)
	columns = int(columns)
	pivot = (int(pivot_x), int(pivot_y))
	if zero_extension.lower() == 'true':
		zero_extension = True
	elif zero_extension.lower() == 'false':
		zero_extension = False
	else:
		raise ValueError(f'Invalid zero_extension on function filter {name}. zero_extension must be either true or false')

	return FUNCTIONS_FILTER_TABLE[name](rows, columns, pivot, zero_extension)
