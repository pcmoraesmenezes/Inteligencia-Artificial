# Machine Learning - Guia de Referência Rápida - Matt Harrison

## 1. Introdução

Esse livro é um guia de referência rápida para Machine Learning. O objetivo é fornecer uma visão geral dos principais conceitos de Machine Learning, com exemplos de código em Python. 

As bibliotecas utilizadas estarão disponíveis no arquivo `utils/requirements.txt`.

## 2. Visão geral do Machine Learning

Uma ótima referência para entender o processo de machine learning é este fluxo de trabalho:

1. Obter dados
2. Limpá-los
3. Explorá-los
4. Modelá-los
5. Avaliar o modelo
6. Repetir

Ou então:

![Fluxo de trabalho](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/visão%20geral%20de%20machine%20learning.jpeg)

## 3. Descrição da classificação: conjunto de dados do Titanic

As etapas aplicadas a um processo de machine learning vão estar melhores descritas com um exemplo. A aplicação pode ser encontrada [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo3.ipynb)

### Termos para os dados

O treinamento de dados é feito através de uma matriz de dados. Cada linha é uma observação (também chamada de exemplo ou registro) e cada coluna é uma característica (também chamada de atributo ou preditor). A última coluna é o rótulo (também chamado de alvo ou classe). O rótulo é o que estamos tentando prever.

Para uma aprendizagem supervisionada, o objetivo é ter uma função que transforme atributos em um rótulo. .

$y = f(X)$

$X$ é uma matriz, onde cada linha representa uma amostra dos dados. Cada coluna em $X$ é um atributo(feature). 

Exemplo:

| Idade | Sexo | Classe | Sobreviveu |
|-------|------|--------|------------|
| 22    | M    | 3      | 0          |
| 38    | F    | 1      | 1          |
| 26    | F    | 3      | 1          |
| 35    | F    | 1      | 1          |

Aqui, temos 4 atributos (os atributos são: idade, sexo, classe e sobreviveu) e 4 amostras (as amostras são as linhas) de dados. Tem-se um rótulo (sobreviveu). Um rótulo é um valor que estamos tentando prever. Neste caso, é um valor binário (0 ou 1).

A saída da função, $y$ é um vetor que contém: rótulos (em uma classificação) ou um valor (em uma regressão).

Exemplo:

| Sobreviveu |
|------------|
| 0          |
| 1          |
| 1          |
| 1          |

Em python, utiliza-se um nome de variável $X$ para armazenar os dados das amostras. O nome da variável $y$ é utilizado para armazenar os rótulos (labels) ou objetivos (targets).

