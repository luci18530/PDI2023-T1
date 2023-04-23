# -*- coding: utf-8 -*-

import json #Importa a biblioteca json para ler arquivos .json
import numpy as np 
from dataclasses import dataclass
from pathlib import Path
from functools import partial
import statistics as stats

def histogram_expansion(arr):
	"""expande o histograma de uma imagem"""
	arr = arr.copy()
	for i in range(arr.shape[-1]):
		band = arr[:,:,i]
		arr[:,:,i] = np.round(((band - np.min(band)) / (np.max(band) - np.min(band))) * 255)
	return arr

# funções que limitam valores de pixels entre 0 e 255
# clip: limita valores entre 0 e 255, valores abaixo de 0 são 0 e valores acima de 255 são 255
# absolute: aplicar valor absoluto a cada pixel, limitando entre 0 e 255
# renormalise: normaliza os valores de cada pixel para o intervalo [0, 255]
LIMIT_FUNCTIONS = {
	'clip': lambda arr: np.clip(arr, 0, 255),
	'absolute': lambda arr:np.clip(np.abs(arr), 0, 255),
	'renormalize': histogram_expansion,
	'abs-renormalize': lambda arr: histogram_expansion(np.abs(arr))
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
	histogram_expansion: bool

    # Define um método que é executado após a inicialização da classe,
	# que checa se o filtro é válido
	def __post_init__(self):
		self.self_check()

	def self_check(self):
		"""checa se o filtro é válido"""
        # Checa se todas as linhas da matriz do kernel têm o mesmo tamanho
		if any(len(row) != len(self.kernel[0]) for row in self.kernel):
			raise ValueError(f'Invalid kernel for filter {self.name}. Not all rows have the same length')

		# if the kernel has only one channel, repeat it 3 times to make it 3 channels
		if len(self.kernel.shape) == 2:
			# self.array = np.repeat(self.array[:, :, np.newaxis], 3, axis=2)
			object.__setattr__(self, "kernel", np.repeat(self.kernel[:, :, np.newaxis], 3, axis=2))
        
        # Checa se o kernel tem pelo menos uma linha e uma coluna
		if all(dim < 1 for dim in self.kernel.shape):
			raise ValueError(f'Invalid kernel for filter {self.name}. Filter must have at least one row and one column')
        
        # Checa se cada "pixel" do kernel tem 1 ou 3 canais
		if any(len(pixel) != 3 for row in self.kernel for pixel in row):
			raise ValueError(f'Invalid kernel for filter {self.name}. Filter "pixels" must have 1 or 3 channels')
        
        # Checa se o pivo tem 2 dimensões
		if len(self.pivot) != 2:
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot must have 2 dimensions')

		# check if pivot elements are integers
		if not all(isinstance(element, int) for element in self.pivot):
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot elements must be integers')

        # Checa se o pivot está dentro dos limites do kernel
		if self.pivot[0] >= len(self.kernel) or self.pivot[1] >= len(self.kernel[0]):
			raise ValueError(f'Invalid pivot for filter {self.name}. Pivot is out of bounds')
        
        # Checa se zero_extension é um booleano
		if not isinstance(self.zero_extension, bool):
			raise ValueError(f'Invalid zero_extension for filter {self.name}. zero_extension must be a boolean')

		# Checa se offset é um inteiro
		if not isinstance(self.offset, int):
			raise ValueError(f'Invalid offset for filter {self.name}. Offset must be an integer')

		# Checa se histogram_expansion é um booleano
		if not isinstance(self.histogram_expansion, bool):
			raise ValueError(f'Invalid histogram_expansion for filter {self.name}. histogram_expansion must be a boolean')

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
        
        # Se o zero_extension for True, a imagem é extendida com zeros para aplicação do filtro
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
        # Se não houver extensão por zeros, a imagem é copiada e as bordas que o pivô não pode passar são removidas
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
				new_img[i, j, :] = self._filter_op(img_padded[row - padding['row']['before']:row + padding['row']['after'] + 1, column - padding['column']['before']:column + padding['column']['after'] + 1]) + self.offset

		filtered_image = self.limit_function(new_img).round().astype(np.uint8)
		if self.histogram_expansion:
			filtered_image = histogram_expansion(filtered_image)

		return filtered_image
	
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
							  filter_json['offset'],
							  filter_json['histogram_expansion'])

	def _filter_op(self, apply_area_array):
		return np.sum(apply_area_array * self.kernel, axis=(0, 1))


@dataclass(frozen=True, init=False)
class FunctionFilter(AbstractFilter):
	"""classe para filtros definidos por uma função"""
	func: callable

	def __init__(self, name, rows, columns, pivot, zero_extension, func):
		super().__init__(name, np.empty((rows, columns, 3)), pivot, zero_extension, LIMIT_FUNCTIONS['clip'], offset=0, histogram_expansion=False)
		object.__setattr__(self, "func", func)

	def _filter_op(self, apply_area_array):
		return self.func(apply_area_array)

# Definindo alguns filtros que serão utilizados posteriormente
BOX_FILTER = partial(FunctionFilter, '[box]', func=lambda array: np.array([np.mean(array[:,:,band]) for band in range(array.shape[2])]))
MEDIAN_FILTER = partial(FunctionFilter, '[median]', func=lambda array: np.array([np.median(array[:,:,band]) for band in range(array.shape[2])]))
MODE_FILTER = partial(FunctionFilter, '[mode]', func=lambda array: np.array([stats.mode(array[:,:,band].flatten()) for band in range(array.shape[2])]))
ERODE_FILTER = partial(FunctionFilter, '[erode]', func=lambda array: np.array([np.min(array[:,:,band]) for band in range(array.shape[2])]))
DILATE_FILTER = partial(FunctionFilter, '[dilate]', func=lambda array: np.array([np.max(array[:,:,band]) for band in range(array.shape[2])]))

# Cria um dicionário que associa uma string (nome do filtro) com uma função de filtro correspondente
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
    
	# Remove os colchetes da string
	filter_str = filter_str[1:-1]
    
    # Separa os parâmetros do filtro por vírgula
	name, rows, columns, pivot_x, pivot_y, zero_extension = filter_str.split(',')
    
    # Verifica se o nome do filtro é válido (está presente na tabela de funções)
	if name not in FUNCTIONS_FILTER_TABLE:
		raise ValueError(f'Invalid function filter name [{name}]')

    # Converte os parâmetros para os tipos corretos
	rows = int(rows)
	columns = int(columns)
	pivot = (int(pivot_x), int(pivot_y))
    
    # Converte o parâmetro zero_extension para booleano
	if zero_extension.lower() == 'true':
		zero_extension = True
	elif zero_extension.lower() == 'false':
		zero_extension = False
	else:
		raise ValueError(f'Invalid zero_extension on function filter {name}. zero_extension must be either true or false')
    
    # Retorna o filtro correspondente à string passada como parâmetro
	return FUNCTIONS_FILTER_TABLE[name](rows, columns, pivot, zero_extension)
