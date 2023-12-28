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

### Crie os atributos

Colunas que não apresentam variação não são úteis para o modelo. Por exemplo, se todos os passageiros do Titanic tivessem a mesma idade, a idade não seria um bom atributo para prever se alguém sobreviveu ou não. 

Como mencionado anteriormente, é necessário descartar leaks de dados, como por exemplo boat e body.

Utiliza-se o método `drop` para remover colunas. O método `drop` retorna um novo DataFrame, portanto, é necessário atribuir o resultado a uma nova variável.

### Separe as amostras de treinamento e teste

Sempre deve-se fazer o treinamento e testes com dados distintos, dessa forma é possível verificar se o modelo está generalizando bem.

Utiliza-se o método `train_test_split` para separar os dados em treinamento e teste. O método `train_test_split` retorna uma tupla com 4 elementos. Os dois primeiros elementos são os dados de treinamento e os dois últimos são os dados de teste.

### Faça a imputação de dados

A imputação de dados é o processo de substituir valores ausentes por valores substitutos.

A coluna `age` possui valores ausentes.  Deve-se imputar os dados de treinamento e teste separadamente, para evitar leaks de dados.

A biblioteca `fancyimpute` possui vários métodos de imputação, entretanto a maioria dos algoritmos não estão implementados de modo indutivo, ou seja não é possível chamar `fit` e `transform` em um conjunto de dados.

De modo semelhante, a classe `InterativeImputer` do `sklearn` também não é indutiva.

A imputação de dados pode ser feita de várias maneiras. Uma maneira simples é substituir os valores ausentes pela média dos valores existentes. Outra maneira é substituir os valores ausentes por um valor constante, como -1.

Para o exemplo é feito de duas maneiras, utilizando a classe `SimpleImputer` do `sklearn` e utilizando a mediana dos dados.

### Normalize os dados

A normalização de dados é o processo de transformar os dados em uma escala comum.

Normalizar os dados ajudará muitos algoritmos de machine learning a convergir mais rapidamente.

Padronizar significa traduzir os dados de modo que eles tenham uma média de 0 e um desvio padrão de 1.

### Pontuação AUC(Area Under the Curve) e ROC(Receiver Operating Characteristic)

#### Curva ROC

A curva ROC mostra o quão bom o modelo criado pode distinguir entre duas coisas. A comparaçõa pode ser realizada entre 0's e 1's ou entre positivo e negativo. Os melhores modelos distinguem com maior precisão esse tipo de comparação.

A curva ROC tem dois parâmetros:

- Taxa de verdadeiro positivo (TPR), ou seja (TP / (TP + FN)), onde TP é verdadeiro positivo e FN é falso negativo.

- Taxa de falso positivo (TNR), ou seja (FP/ (FP+TN)) onde FP é falso positivo e TN é verdadeiro negativo.

- A curva ROC é um gráfico de TPR vs FPR.

#### Curva AUC

A curva auc é uma derivada da curva ROC.

O valor do AUC varia de 0,0 até 1,0

Quanto maior o AUC melhor!

Exemplo: 

![Curva ROC E AUC](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/curva%20roc%20e%20auc.png)


Um modelo com previsão 100% errada tem um AUC de 0.0, enquanto um modelo que tem previsão 100% correta em um AUC de 1.0.

Cada modelo possuí um AUC!

### Validação cruzada (k-fold cross-validation)

A validação cruzada é uma técnica para avaliar modelos de machine learning, treinando vários modelos em subconjuntos dos dados disponíveis e avaliando-os nos subconjuntos retidos.

Na validação cruzada k-fold, os dados são divididos em k subconjuntos. Um modelo é treinado usando k-1 subconjuntos e o outro subconjunto é usado para testar o modelo. Isso é repetido para cada subconjunto e, em seguida, a média dos resultados é calculada.

Um modelo com uma pontuação média mais alta é melhor do que um modelo com uma pontuação média mais baixa.

Entretanto, um modelo com uma pontuação média um pouco menor, mas com um désvio padrão menor, pode ser melhor do que um modelo com uma pontuação média mais alta, mas com um désvio padrão maior.

### Stack de modelos

Um classificador de stack utiliza a saída de vários modelos como entrada para um modelo final. O modelo final é chamado de meta-modelo.

A stack de modelos pode ser utilizada para melhorar a precisão de um modelo.

Nem sempre a stack de modelos melhora a precisão do modelo.

### Hiperparâmetros

Os hiperparâmetros são parâmetros que não são aprendidos pelo modelo. Eles são definidos antes do treinamento do modelo.

```python

random_forest2 = ensemble.RandomForestClassifier()
params = {
    "max_features": [0.4, "auto"],
    "n_estimators": [15,200],
    "min_samples_leaf": [1, 0.1],
    "random_state": [42],
}

```

Pode-se utilizar o método `GridSearchCV` para encontrar os melhores hiperparâmetros.

### Matriz de confusão

A matriz de confusão é uma tabela que mostra as frequências de classificação para cada classe de um modelo.

Através dela é possível verificar a quantidade de falsos positivos e falsos negativos.

### Curva de aprendizado

A curva de aprendizado é uma ferramenta para ver como o modelo está aprendendo. A curva de aprendizado mostra a pontuação de treinamento e a pontuação de validação para diferentes tamanhos de conjunto de treinamento.

É possível verificar se o modelo está sofrendo de overfitting ou underfitting.

Também é possível verificar se os dados de treinamento são suficientes para o modelo.

## Capítulo 4 - Dados Ausentes

O tratamento de dados ausentes deve ser feito, pois a maioria dos algoritmos de machine learning não aceitam dados ausentes, salve algumas exceções como os algoritmos de XGBoost.

Além disso, não existe um melhor mecanismo para se lidar com dados ausentes, pois cada conjunto de dados é diferente. O principal seria identificar porque os dados estão ausentes, por exemplo, se os dados estão ausentes porque o usuário não preencheu o formulário, então é possível preencher os dados ausentes com a média dos dados existentes, entretanto se o dado está ausente porque o usuário não preencheu o formulário, mas o dado é importante, então é melhor remover a linha.

Existem diversas maneiras de se lidar com dados ausentes, como por exemplo:

- Remover a linha

- Remover a coluna

- Fazer a imputação de dados (substituir os dados ausentes por um valor substituto)

Vamos dar uma olhada na prática, acesse aqui o [código](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo4.ipynb)

A prática contém diversos gráficos para exibição de dados ausentes que pode ser extremamente útil para a análise exploratória de dados.

Além disso nela contém informações sobre como lidar com dados ausentes em dados categóricos e numéricos.

### Imputação de Dados

A imputação de dados é o processo de substituir valores ausentes por valores substitutos. Utilizamos depois de termos feito o modelo de predição.

A imputação de dados requer que os dados de treinamento e teste sejam imputados separadamente, para evitar leaks de dados.

É necessário construir um pipeline para imputar os dados de treinamento e teste.

Mais informações disponíveis na prática [código](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo4.ipynb)



## Capítulo 5 - Fazendo uma Limpeza nos Dados

A limpeza de dados é o processo de transformar dados brutos em dados limpos. Os dados brutos podem ser dados que foram coletados de uma fonte externa, como um banco de dados ou um arquivo CSV. 

A limpeza de dados é um processo iterativo. É necessário verificar os dados, fazer alterações e verificar novamente os dados.

A limpeza de dados pode ser feita de várias maneiras, como por exemplo:

- Remover dados duplicados

- Remover dados inconsistentes (por exemplo, um valor de idade de -1)

- Remover dados irrelevantes

- Remover dados que vazam (por exemplo, dados que não estariam disponíveis no momento da previsão)

- Remover dados que não são necessários (por exemplo, dados que não são necessários para o modelo)

- Remover outliers

- Lidar com dados ausentes

No geral a limpeza de dados é extremamente necessária, principalmente ao ler o capítulo 4, foi visto que a maioria dos algoritmos de machine learning não aceitam dados ausentes, então é necessário fazer a limpeza dos dados.

Além disso a limpeza de dados está além do que simplesmente lidar com dados ausentes, pode ser feita uma melhora na forma como os dados estão sendo exibidos para assim, transmitir uma clareza maior sobre os dados. Como por exemplo, remover espaços em branco, remover caracteres especiais.

Na prática desta seção estará as informações sobre como fazer a limpeza de dados, acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo5.ipynb)

## Capítulo 6 - Explorando os Dados

Antes de criar o modelo de machine learning, é necessário explorar os dados. A exploração de dados é o processo de descobrir informações sobre os dados. Uma excelente forma de explorar os dados é através da análise exploratória de dados. Essa forma garante uma noção maior sobre os dados e descobrir insights que podem ser úteis para o modelo.

