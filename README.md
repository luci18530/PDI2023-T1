# Sistema para processamento de imagens RGB

<p align="justify"> Este repositório contém o código-fonte de um sistema para abrir, exibir, manipular e salvar imagens RGB com 24 bits/pixel (8 bits/componente/pixel) na linguagem de programação Python. O sistema foi desenvolvido como trabalho prático para a disciplina de Introdução ao Processamento Digital de Imagens ministrada pelo professor Leonardo Vidal Batista. </p>

## Funcionalidades

1. Conversão RGB-YIQ-RGB.

2. Negativo em RGB (banda a banda) e na banda Y, com posterior conversão para RGB.

3. <p align="justify"> Correlação m x n (inteiros não negativos), com extensão por zeros, sobre R, G e B, com offset (inteiro) e filtro definidos em um arquivo (json) à parte. Filtros Soma, Box, ||Sobel|| e Emboss foram testados e os resultados explicados. Também foi comparado Box11x1(Box1x11(Image)) com Box(11x11), em termos de resultado e tempo de processamento. Para o Sobel, foi aplicada expansão de histograma para [0, 255]. Para o filtro de Emboss, foi aplicado valor absoluto ao resultado da correlação, e então somado o offset. </p>

4. Filtro mediana m x n, com m e n ímpares, sobre R, G e B.

## Como usar

A ser feito

## Autores

- Lucas Issac
- Luciano Pereira

