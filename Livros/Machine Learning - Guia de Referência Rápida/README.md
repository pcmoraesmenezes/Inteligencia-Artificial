# Machine Learning - Guia de Referência Rápida - Matt Harrison

## Tabela de Conteúdos

- [1. Introdução](#1-introdução)
- [2. Visão geral do Machine Learning](#2-visão-geral-do-machine-learning)
- [3. Descrição da classificação: conjunto de dados do Titanic](#3-descrição-da-classificação-conjunto-de-dados-do-titanic)
  - [Termos para os dados](#termos-para-os-dados)
  - [Crie os atributos](#crie-os-atributos)
  - [Separe as amostras de treinamento e teste](#separe-as-amostras-de-treinamento-e-teste)
  - [Faça a imputação de dados](#faça-a-imputação-de-dados)
  - [Normalize os dados](#normalize-os-dados)
  - [Pontuação AUC(Area Under the Curve) e ROC(Receiver Operating Characteristic)](#pontuação-aucarea-under-the-curve-e-rocreceiver-operating-characteristic)
    - [Curva ROC](#curva-roc)
    - [Curva AUC](#curva-auc)
  - [Validação cruzada (k-fold cross-validation)](#validação-cruzada-k-fold-cross-validation)
  - [Stack de modelos](#stack-de-modelos)
  - [Hiperparâmetros](#hiperparâmetros)
  - [Matriz de confusão](#matriz-de-confusão)
  - [Curva de aprendizado](#curva-de-aprendizado)
  - [Código Fonte](#código-fonte)
- [Capítulo 4 - Dados Ausentes](#capítulo-4---dados-ausentes)
    - [Imputação de Dados](#imputação-de-dados)
    - [Código Fonte](#código-fonte-1)
- [Capítulo 5 - Fazendo uma Limpeza nos Dados](#capítulo-5---fazendo-uma-limpeza-nos-dados)
    - [Código Fonte](#código-fonte-2)
- [Capítulo 6 - Explorando os Dados](#capítulo-6---explorando-os-dados)
    - [Código Fonte](#código-fonte-3)
- [Capítulo 7 - Pré Processamento de Dados](#capítulo-7---pré-processamento-de-dados)
    - [Padronização de Dados](#padronização-de-dados)
    - [Escale para um intervalo](#escale-para-um-intervalo)
        - [Explicação do calculo de escalonamento de dados.](#explicação-do-calculo-de-escalonamento-de-dados)
        - [Explicação por trás do calculo de escalonamento de dados.](#explicação-por-trás-do-calculo-de-escalonamento-de-dados)
    - [Variáveis Dummy](#variáveis-dummy)
    - [Codificação de Rótulos](#codificação-de-rótulos)
    - [Codificador de Frequência](#codificador-de-frequência)
    - [Extraindo categorias a partir de strings](#extraindo-categorias-a-partir-de-strings)
    - [Outras codificações](#outras-codificações)
    - [Engenharia de Dados para Datas](#engenharia-de-dados-para-datas)
    - [Adição do atributo col_na](#adição-do-atributo-col_na)
    - [Engenharia de dados manual](#engenharia-de-dados-manual)
    - [Código Fonte](#código-fonte-4)
- [Capítulo 8 - Seleção de Atributos](#capítulo-8---seleção-de-atributos)
    - [A Maldição da Dimensionalidade](#a-maldição-da-dimensionalidade)
    - [Colunas colineares](#colunas-colineares)
    - [Regressão Lasso](#regressão-lasso)
    - [Eliminação Recursiva de atributos](#eliminação-recursiva-de-atributos)
    - [Informações Mútuas](#informações-mútuas)
    - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
    - [Importância dos Atributos](#importância-dos-atributos)
    - [Código Fonte](#código-fonte-5)
- [Classes Desbalanceadas](#classes-desbalanceadas)
    - [Use métricas diferentes](#use-métricas-diferentes)
    - [Algoritmos baseados em árvore e Ensemble](#algoritmos-baseados-em-árvore-e-ensemble)
    - [Modelos de penalização](#modelos-de-penalização)
    - [Gerando dados de minorias](#gerando-dados-de-minorias)
    - [Upsampling e depois downsampling](#upsampling-e-depois-downsampling)
    - [Código Fonte](#código-fonte-6)
- [Capítulo 10 - Classificação](#capítulo-10---classificação)
    - [Regressão Logística](#regressão-logística)
        - [Parâmetros](#parâmetros)
    - [Naive Bayes](#naive-bayes)
        - [Eficiência na Execução](#eficiência-na-execução)
        - [Pré-processamento dos Dados](#pré-processamento-dos-dados)
        - [Para evitar superadequação](#para-evitar-superadequação)
        - [Interpretação dos Resultados](#interpretação-dos-resultados)
        - [Parâmetros da instância](#parâmetros-da-instância)
        - [Aviso](#aviso)
    - [Máquinas de Vetores de Suporte](#máquinas-de-vetores-de-suporte-svm)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-1)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-1)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-1)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-1)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-1)
    - [K-Nearest Neighbors](#k-vizinhos-mais-próximos)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-2)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-2)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-2)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-2)
        - [Atributos](#atributos)
        - [Atenção](#atenção)
    - [Árvores de Decisão](#árvores-de-decisão)
        - [Atenção](#atenc3a7c3a3o-1)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-3)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-3)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-3)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-3)
        - [Atributos](#atributos-1)
    - [Florestas Aleatórias](#florestas-aleatórias)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-4)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-4)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-4)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-4)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-2)
        - [Atributos-pós adequação](#atributos-apc3b3s-a-adequac3a7c3a3o-1) 
    - [XGBoost](#xgboost)  
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-5)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-5)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-5)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-5)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-3)
        - [Atributos-pós adequação](#atributos-2)
    - [LightGBM](#gradient-boosting-com-lightgbm)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-6)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-6)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-6)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-6)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-4)
    - [TPOT](#tpot)
        - [Eficiência na Execução](#eficic3aancia-na-execuc3a7c3a3o-7)
        - [Pré-processamento dos Dados](#prc3a9-processamento-dos-dados-7)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-7)
        - [Interpretação dos Resultados](#interpretac3a7c3a3o-dos-resultados-7)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-5)
        - [Atributos-pós adequação](#atributos-3)
    - [Código Fonte](#códigos)
- [Capítulo 11 - Seleção de Modelo](#capítulo-11---seleção-do-modelo)
    - [Validation Curve](#curva-de-validação)
    - [Learning Curve](#curva-de-aprendizagem)
    - [Código Fonte](#cc3b3digos-1)

- [Capítulo 12 - Métricas e avaliação de classificação](#capítulo-12---métricas-e-avaliação-de-classificação)
    - [Matriz de Confusão](#matriz-de-confusc3a3o-1)
    - [Métricas](#métricas)
        - [Acurácia](#acurácia)
        - [Recall](#recall)
        - [Precisão](#precisão)
        - [F1](#f1)
    - [Relatório de classificação](#relatório-de-classificação)
    - [Curva ROC](#roc-receiver-operating-characteristic)
    - [Curva precisão recall](#curva-de-precisão-recall)
    - [Gráfico de elevação](#gráfico-de-elevação)
    - [Balanceamento das classes](#balanceamento-das-classes)
    - [Erro de predição de classe](#erro-de-predição-de-classe)
    - [Limiar de discriminação](#limiar-de-discriminação)
    - [Código fonte](#cc3b3digos-2)
- [Capítulo 13 - Explicando os Modelos](#capítulo-13---explicando-os-modelos)
    - [Coeficientes de Regressão](#coeficientes-de-regressão)
    - [Importância dos Atributos](#importc3a2ncia-dos-atributos-1)
    - [LIME](#lime-local-interpretable-model-agnostic-explanations)
    - [Interpretação de árvores](#interpretação-de-árvores)
    - [Gráficos de dependência parcial](#gráficos-de-dependência-parcial)
    - [Modelos Substitutos](#modelos-substitutos)
    - [SHAP](#shap-shapley-additive-explanations)
    - [Código Fonte](#cc3b3digos-3)
- [Capítulo 14 - Regressão](#capítulo-14---regressão)
    - [Modelo de base](#modelo-de-base)
    - [Regressão Linear](#regressão-linear)
        - [Eficiência na execução](#eficic3aancia-na-execuc3a7c3a3o-8)
        - [Pré processamento dos dados](#prc3a9-processamento-dos-dados-8)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-8)
        - [Interpretação dos resultados](#interpretac3a7c3a3o-dos-resultados-8)
        - [Parâmetros de instância](#parc3a2metros-da-instc3a2ncia-6)
        - [Atributos após a adequação](#atributos-apc3b3s-a-adequac3a7c3a3o-2)
    - [SVMs](#svms)
        - [Eficiência na execução](#eficic3aancia-na-execuc3a7c3a3o-9)
        - [Pré processamento dos dados](#prc3a9-processamento-dos-dados-9)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-9)
        - [Interpretação dos resultados](#interpretac3a7c3a3o-dos-resultados-9)
        - [Parâmetros de instância](#parc3a2metros-da-instc3a2ncia-7)
        - [Atributos após a adequação](#atributos-apc3b3s-a-adequac3a7c3a3o-3)
    - [K-Nearest Neighboors](#k-nearest-neighbors)
        - [Eficiência na execução](#eficic3aancia-na-execuc3a7c3a3o-10)
        - [Pré processamento dos dados](#prc3a9-processamento-dos-dados-10)
        - [Para evitar superadequação](#para-evitar-superadequac3a7c3a3o-10)
        - [Interpretação dos resultados](#interpretac3a7c3a3o-dos-resultados-10)
        - [Parâmetros da instância](#parc3a2metros-da-instc3a2ncia-8)

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

### Código Fonte

O código fonte pode ser encontrado [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo3.ipynb)

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


### Código Fonte
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

### Código Fonte

Na prática desta seção estará as informações sobre como fazer a limpeza de dados, acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo5.ipynb)

## Capítulo 6 - Explorando os Dados

Antes de criar o modelo de machine learning, é necessário explorar os dados. A exploração de dados é o processo de descobrir informações sobre os dados. Uma excelente forma de explorar os dados é através da análise exploratória de dados. Essa forma garante uma noção maior sobre os dados e descobrir insights que podem ser úteis para o modelo.

Esse capítulo foi extremamente prático, pois foi utilizado de ferramentas visuais para a exploração de dados, como por exemplo, gráficos de barras, gráficos de pizza, gráficos de dispersão, gráficos de caixa, histogramas, etc.

### Código Fonte
Para a visualização da prática acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo6.ipynb)

## Capítulo 7 - Pré Processamento de Dados

### Padronização de Dados

Alguns algoritmos, como o SVM, são sensíveis à escala dos dados, ou seja, apresentam melhores desenvolvimentos quando os dados estão na mesma escala. Dados padronizados implica em dados com média 0 e desvio padrão 1.

### Escale para um intervalo

Escalonar os dados para um intervalo é uma técnica para transformar os dados para que eles estejam entre um intervalo específico. Por exemplo, pode-se transformar os dados para que eles estejam entre 0 e 1. Para valores discrepantes, deve-se ter cuidado ao usar essa técnica.

#### Explicação do calculo de escalonamento de dados.

A formula para o calculo de escalonamento de dados é:

$$ x_{normalizado} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

Por exemplo, imagine que você tenha um conjunto de dados com os seguintes valores:

$$ x = [2, 5, 10, 20, 30] $$

$$ x_{min} = 2 $$
$$ x_{max} = 30 $$

$$ x_{normalizado} = \frac{[2, 5, 10, 20, 30] - 2}{30 - 2}$$

$$ x_{normalizado} = \frac{[0, 3, 8, 18, 28]}{28}$$

$$ x_{normalizado} = [0, 0.107, 0.286, 0.643, 1]$$

#### Explicação por trás do calculo de escalonamento de dados.

O escalonamento de dados é uma técnica de pré-processamento de dados que é usada para padronizar os dados de entrada.

$ X - X_{min} $ É a distância do valor original até o valor minimo. Essa parte assegura que o menor valor seja 0.

$ X_{max} - X_{min} $ É a distância entre o valor máximo e o valor mínimo. Essa parte assegura que o maior valor seja 1.

Dividir a distância do valor original até o valor mínimo pela distância entre o valor máximo e o valor mínimo, assegura que os valores estejam entre 0 e 1.

### Variáveis Dummy

Variáveis dummy são variáveis categóricas que são transformadas em variáveis numéricas. Por exemplo, a variável categórica sexo pode ser transformada em duas variáveis numéricas: sexo masculino e sexo feminino. Se o sexo for masculino, a variável sexo masculino será 1 e a variável sexo feminino será 0. Se o sexo for feminino, a variável sexo masculino será 0 e a variável sexo feminino será 1.

Esse procedimento é chamado de one-hot encoding.

### Codificação de Rótulos

É uma técnica alternativa para as variaveis dummy. Nesse caso, cada categoria é atribuída a um numero. É extremante útil para variáveis com muitas categorias. Esse método impõe uma ordem nas categorias, o que pode não ser desejável. Alguns algoritmos como os em árvore são capazes de lidar com essa codificação.

O codificador de rotulos consegue lidar com uma coluna por vez

### Codificador de Frequência

Outra opção viável para variáveis com alta cardinalidade. Esta técnica implica em substituir cada categoria pelo número de vezes em que ela aparece.  Pode ser utilizado o pandas para fazer essa codificação.

Lembre-se de armazenar o mapa de codificação para que seja possível fazer a codificação dos dados de teste.

### Extraindo categorias a partir de strings

Este método é útil e simples e pode aumentar a precisão dos modelos de machine learning. Ele envolve a extração de categorias de uma string. Por exemplo, se você tiver uma coluna de endereço, poderá extrair o estado e o país dessa coluna. 

### Outras codificações

A biblioteca `category_encoders` possui outras codificações que podem ser úteis para a criação de modelos de machine learning. Essa biblioteca retorna um DataFrame, o que, de modo geral, é mais fácil de trabalhar do que um array NumPy.

Um algoritmo útil dessa biblioteca é o codificador de hash, principalmente quando não se sabe o número de categorias que uma variável pode ter.

### Engenharia de Dados para Datas

A biblioteca `fastai` possui uma classe `add_datepart` que pode ser usada para gerar novas colunas de data a partir de uma coluna de data existente. Essa classe pode ser usada para gerar novas colunas de data, como dia da semana, dia do ano, etc. Isso é útil, pos a maioria dos modelos de machine learning não aceitam dados de data.

### Adição do atributo col_na

Saber que um valor estava ausente pode ser útil para um modelo de machine learning. Por exemplo, se o valor de uma variável for ausente, o modelo pode atribuir um valor diferente a essa variável.

Pode ser interessante adicionar uma coluna que indique se o valor estava ausente ou não.

### Engenharia de dados manual

O `pandas` pode ser utilizado para gerar novos atributos. Por exemplo, pode-se criar uma coluna que indique se o passageiro é criança ou adulto.Para obter dados agregados de um DataFrame, pode-se usar o método `groupby`. E para mesclar os dados agregados de volta ao DataFrame original, pode-se usar o método `merge`.

### Código Fonte

Para a visualização da prática acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo7.ipynb)

## Capítulo 8 - Seleção de Atributos

A feature selection (seleção de atributos) é o processo de selecionar os atributos úteis ao model. Atributos irrelevantes e atributos correlacionados podem causar um efeito negativo ao modelo, podendo até mesmo deixar os coeficientes do modelo instáveis.

### A Maldição da Dimensionalidade

A maldição da dimensionalidade (curse of dimensionality) é um problema que ocorre quando os dados possuem muitas dimensões. Esse problema pode causar overfitting e tornar o modelo lento.

A medida que o número de dimensões aumenta, o número de amostras necessárias para preencher o espaço aumenta exponencialmente. Cálculos de distância também se tornam mais difíceis com o aumento do número de dimensões.

Com isso, o tempo de treinamento dos dados, que é uma função do número de colunas, aumenta exponencialmente. Tendo isso em mente, a seleção correta de atributos pode ser extremamente útil no quesito performance.

### Colunas colineares

Colunas colineares são colunas que estão altamente correlacionadas. Essas colunas podem causar instabilidade nos coeficientes do modelo. Além disso, elas não adicionam informações úteis ao modelo.

Algumas funções para checar as colunas foram apresentadas, como por exemplo, a função `corr` do `pandas` e a função `heatmap` do `seaborn`.

O `Rank2` do `yellowbrick` pode ser usado afim de verificar a correlação entre as colunas.

O pacote `rfpimp` tem um recurso de visualização de multicolinearidade.

A recomendação nestes gráficos é encontrar valores bem próximos de 1.

### Regressão Lasso

A regressão Lasso pode ser usada para definir um parametro alpha que atuará como um parametro de regularização. A medida que seu valor aumenta, menor é o peso.

### Eliminação Recursiva de atributos

A eliminação recursiva remove os atributos mais fracos e ajusta os modelos.

### Informações Mútuas

A informação mútua é uma medida de dependência entre duas variáveis. Ela mede a quantidade de informação que uma variável contém sobre a outra. 

O `sklearn` possui uma função `mutual_info_classif` que pode ser usada para calcular a informação mútua entre duas variáveis. Os testes são feitos utilizando os k vizinhos mais próximos.

### Principal Component Analysis (PCA)

O PCA é uma técnica de redução de dimensionalidade. Ele reduz o número de dimensões de um conjunto de dados, mantendo o máximo de informações possível. O PCA é uma técnica não supervisionada. 

O PCA pode ser usado para reduzir o número de dimensões de um conjunto de dados. Isso pode ser útil para reduzir o tempo de treinamento do modelo.

### Importância dos Atributos

A importância dos atributos é uma medida da importância de cada atributo para o modelo. Essa medida pode ser usada para selecionar os atributos mais importantes.

A maioria dos modelos baseados em árvore possui um atributo `feature_importances_` que pode ser usado para calcular a importância dos atributos.

### Código Fonte

Para a visualização da prática acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo8.ipynb)

## Classes Desbalanceadas

Em casos de classificação pode ocorrer de encontrar classes desbalanceadas, ou seja, uma classe pode ter uma quantidade muito maior de amostras do que a outra. Isso pode causar um problema, pois o modelo pode aprender a prever apenas a classe com maior quantidade de amostras. Por exemplo, se tiver um caso positivo e 99 casos negativos, o modelo poderá ter 99% de precisão, prevendo apenas a classe negativa.

### Use métricas diferentes

A precisão não é uma boa métrica para classes desbalanceadas. Nesse caso, é melhor usar outras métricas, como a pontuação F1, a área sob a curva ROC (AUC) ou a precisão média de precisão (AP).

### Algoritmos baseados em árvore e Ensemble

Modelos baseados em árvore tendem a ter um melhor desempenho conforme a distribuição da classe menor. Além disso caso os dados estejam agrupados, podem ser utilizados algoritmos de ensemble, como o XGBoost.

### Modelos de penalização

A vasta maioria dos modelos do `scikit-learn`aceitam o parâmetro `class_weight`. Defini-lo como `balanced` tentará regularizar as classes mais desbalanceadas. Uma alternativa seria o `gridsearch` para encontrar o melhor valor para o parâmetro.

A biblioteca `XGBoost` possuí um parametro `max_delta_step` que pode ser utilizado para penalizar o modelo. Além disso, parametros como `scale_pos_weight` define a razão entre as classes e `eval_metric` define a métrica de avaliação, que pode ser substituída por `auc` ou `aucpr`.

O modelo KNN, tem o parâmetro `weights` que pode ser utilizado para penalizar o modelo. Além disso, `distance` pode ser utilizado para penalizar o modelo de acordo com a distância. Issso pode ser útil para classes desbalanceadas.

### Gerando dados de minorias

A biblioteca `imbalanced-learn` possui algumas tecnicas de amostragem que podem ser utilizadas para gerar dados de minorias. Por exemplo, o `RandomOverSampler` pode ser utilizado para gerar dados de minorias aleatórios. O `SMOTE` pode ser utilizado para gerar dados de minorias sintéticos. O `SMOTE` funciona selecionando uma amostra da classe minoritária e calculando os k vizinhos mais próximos para essa amostra. Em seguida, ele seleciona um dos vizinhos aleatoriamente e calcula um ponto intermediário entre a amostra e o vizinho. Esse ponto intermediário é um novo ponto de dados sintético.

### Upsampling e depois downsampling

A biblioteca `imbalanced-learn` implementa `SMOTEENN` e `SMOTETomek`, que são combinações de upsampling e downsampling. É inicialmente  aplicado um upsampling e depois um downsampling, afim de limpar os dados.

### Código Fonte

Para a visualização da prática acesse [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo9.ipynb)

## Capítulo 10 - Classificação

A classificação é um método de aprendizagem supervisionada para atribuir rótulo a uma amostra com base nos atributos. Por supervisionada, implica que haverá ou rotulos para a classificação; ou numeros para a regressão.

O `sklearn` implementa diversos modelos úteis de classificação. Além disso possuem interfaces consistentes, o que facilita a utilização.

No `sklearn`, cria-se uma instância de modelo e chama-se o método `fit` para treinar o modelo. Em seguida, chama-se o método `predict` para fazer previsões. Pode ser utilizado o `.score`` para verificar a precisão do modelo.

O grande problema deste tipo de modelo é organizar os dados em um formato apropriado para o `Sklearn`. Os dados devem estar em um array $M$ com dimensão de $M_{n\dot m}$ ou em um DataFrame do `pandas`. O rótulo (target) deve ser um array $M$ com dimensão de $M_{n}$.

Um outro problema é o método `.score`, ele por si só pode não ser suficiente para avaliar o modelo. É necessário utilizar outras métricas, como por exemplo, a matriz de confusão, a curva ROC e a curva AUC.

Alguns dos métodos genéricos do `sklearn` são:

- `fit(X, y[, sample_weight])` - Ajusta o estimador aos dados de treinamento.

- `predict(X)` - Preveja os rótulos para X.

- `predict_log_proba(X)` - Preveja o log-probabilidade de classes.

- `predict_proba(X)` - Preveja probabilidades de classe para X.

- `score(X, y[, sample_weight])` - Retorna a precisão média nos dados de teste e rótulos fornecidos.

### Regressão Logística

Apesar de possuir regressão no nome, ela não é uma técnica de regressão, mas sim de classificação. ELa estima as probabilidades usando uma função logística. Esse modelo é padrão para a maioria dos problemas de classificação.

#### Parâmetros

Eficiência na execução:

- Pode ser usado o `n_jobs` caso não esteja utilizando o solucionador `liblinear`.

Pré-processamento dos dados:
Se `solver` estiver definido com `sag` ou `saga`, padronize para que a convergência funcione. É capaz de lidar com entradas esparsas.

Para evitar superadequação:

O parâmetro `C` controla a regularização. (Valores menores especificam uma regularização mais forte). É possível especificar `penalty` com `l1` ou `l2`

Interpretação dos resultados:

O atributo `.coef` do modelo apos a adequação mostra os coeficientes da função de decisão.

Parâmetros da instância:
`penalty` - Especifica a norma usada na penalidade. `l1` ou `l2` Padrão é `l2`.

`dual` - Seleciona o algoritmo de otimização. `False` quando `n_samples > n_features`. Padrão é `False`.

`tol` - Tolerância para critério de parada. Padrão é `1e-4`.

`C` - Inverso da força de regularização. Padrão é `1.0`. Um valor menor especifica uma regularização mais forte.

`fit_intercept` - Especifica se uma constante (viés) deve ser adicionada à função de decisão. Padrão é `True`.

`intercept_scaling` - Quando `solver` é `liblinear` e `fit_intercept` é `True`, `intercept_scaling` especifica a escala do intercepto. Padrão é `1`.

`class_weight` - Define o peso das classes. Padrão é `None`.

`max_iter` - Número máximo de iterações. Padrão é `100`.

`multi_class` - Especifica como o modelo deve lidar com uma classificação multiclasse. `ovr` ou `multinomial`. Padrão é `ovr`.

`verbose` - Define o nível de verbosidade. Padrão é `0`.

`warm_start` - Quando definido como `True`, reutiliza a solução da chamada anterior para ajustar como inicialização. Padrão é `False`.

`solver` - Especifica o algoritmo de otimização. `newton-cg`, `lbfgs`, `liblinear`, `sag`, `saga`. Padrão é `liblinear`. `liblinear` para pequenos conjuntos de dados. `saga` `lbfgs` para dados de multiclasse. 

`coef_` - Coeficientes da função de decisão.

`njobs` - Número de trabalhos em paralelo. Padrão é `None`.

`intercept_` - Constante (viés) adicionada à função de decisão.

`n_iter_` - Número real de iterações para alcançar a convergência. Padrão é `None`.

O intercepto é o log odds da condição de base. É possível converter o intercepto em probabilidades usando a função logística.

```python
def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))`
```

O log odds é a probabilidade de um evento ocorrer dividido pela probabilidade de um evento não ocorrer.

É possível inspecionar os coeficientes para ver quais atributos são mais importantes para o modelo.

### Naive Bayes

O naive Bayes é um classificador probabilistico que pressupõe uma independência entre os atributos. Ele é rápido e simples, usualmente utilizado para identificação de spam e classificação de texto. Uma das grandes antagens desse modelo é que, por supor uma indepência entre os atributos, ele é capaz de fazer o treinamento de um modelo com um numero pequeno de amostras. Entretanto uma desvantagem é que ele não consegue captar as iterações entre os atributos. Ele também é bom para dados com muitos atributos.

Existem três tipos de naive Bayes no `sklearn`:

`GaussianNB` - Assume que os dados seguem uma distribuição normal e atributos continuos.

`BernoulliNB` - Assume que os dados seguem uma distribuição de Bernoulli e atributos binários.

`MultinomialNB` - Assume que os dados seguem uma distribuição multinomial e atributos discretos.

#### Eficiência na execução

Treinamento O(Nd) em que N é o número de exemplos para treinamento e d e a dimensão dos dados.

#### Pré-processamento dos dados

É pressuposto que os dados sejam independentes. O desempenho é aprimorado com a remoção de atributos correlacionados.

#### Para evitar superadequação

O parâmetro `alpha` controla a regularização. Um valor menor especifica uma regularização mais forte.

#### Interpretação dos resultados

A porcentagem é a probabilidade de uma amostra pertencer a uma classe específica.

#### Parâmetros da instância

`priors` - Probabilidades a priori de cada classe. Padrão é `None`.

`var_smoothing` - Adiciona um valor para a variancia dos atributos. Padrão é `1e-9`.

`class_prior` - Probabilidades a priori de cada classe. Padrão é `None`.

`class_count_` - Número de amostras em cada classe.

`theta_` - Média de cada atributo por classe.

`sigma_` - Variância de cada atributo por classe.

`epsilon_` - Valor adicionado à variancia dos atributos.

#### Aviso:

Esse tipo de modelo são suscetivos ao problema da probabilidade zero. Caso tente classificar uma nova amostra que não tenha sido vista no treinamento, o modelo irá retornar uma probabilidade zero. Para evitar esse problema, pode ser utilizado o `Laplace smoothing`, que adiciona um valor a variancia dos atributos. O `sklearn` implementa esse método através do parâmetro `var_smoothing`.

### Máquinas de Vetores de Suporte (SVM)

As máquinas de vetores de suporte são um modelo de classificação que encontra um hiperplano que separa os dados. Esse modelo é útil para dados com muitas dimensões. Ele também é útil para dados que não são linearmente separáveis, pois é possível usar um kernel para transformar os dados em um espaço de dimensão superior. Os vetores de suporte são os pontos de dados mais próximos do hiperplano. O hiperplano é definido por um vetor de pesos e um viés. O vetor de pesos é perpendicular ao hiperplano. O viés é o deslocamento do hiperplano da origem.

Existem algumas implementações de SVM no `sklearn`, como por exemplo `SVC` encapsula a biblioteca `libsvm` e `LinearSVC` encapsula a biblioteca `liblinear`. Também há o `linear_model.SGDClassifier` que implementa o SVM com gradiente descendente estocástico.

Em geral, a SVM tem um bom desempenho e oferece suporte para espaços lineares e não lineares usando truques de kernel. Entretanto, o treinamento pode ser lento para grandes conjuntos de dados. Além disso, é difícil interpretar os resultados. O kernel default é o `Radial Basis Function 'rbf' `, controlado pelo padrão `gamma`, o que permite mapear um espaço de entrada em um espaço com mais dimensões.

#### Eficiência na execução

A implementação do `scikit-learn` é $O(n^{4})$, o que pode ser dificil para tamanhos de dados grandes. Utilizando um kernel linear é possível utilizar o `LinearSVC` que é $O(n^{3})$. É válido mencionar que o desempenho é acompanhado da perda de precisão.

#### Pré-processamento dos dados

É necessário padronizar os dados para que o desempenho seja melhor. É possível utilizar o `StandardScaler` do `sklearn` para padronizar os dados.

#### Para evitar superadequação

O parâmetro `C` controla a regularização. Um valor menor especifica uma regularização mais forte. Um valor maior para `gamma` tenderá a uma superadequação. O modelo `LinearSVC` aceita parâmetros `penalty` e `loss` que podem ser utilizados para regularização.

#### Interpretação dos resultados

O atributo `coef_` do modelo após o ajuste mostra os coeficientes da função de decisão.

Para obter probabilidades use `probability=True` ao criar o modelo. Isso pode ser útil para a calibração de probabilidades, entretanto, pode ser lento.

#### Parâmetros da instância

`C` - Parâmetro de regularização. Padrão é `1.0`, esse parametro de penalidade quanto menor, maior a fronteira para a superadequação.

`kernel` - Especifica o kernel a ser utilizado. `linear`, `poly`, `rbf`, `sigmoid`, `precomputed` ou `callable`. Padrão é `rbf`.

`cache_size` - Especifica o tamanho do cache em MB. Padrão é `200`. Aumentar esse valor pode ser útil em grandes conjuntos de dados.

`class_weight` - Define o peso das classes. Padrão é `None`. Use dictionary para definir pesos diferentes para cada classe.

`coef0` - Termo independente para kernels polinomiais e sigmoides. Padrão é `0.0`.

`decision_function_shape` - Especifica a forma da função de decisão. `ovo` ou `ovr`. Padrão é `ovr`. `ovo`(one-vs-one) cria um classificador binário para cada par de classes. `ovr`(one-vs-rest) cria um classificador binário para cada classe.

`degree` - Grau do kernel polinomial. Padrão é `3`.

`gamma` - Coeficiente do kernel. Pode ser um número, `scale`default em $0.22),1/(num atributos * X.std())$. Um valor menor resulta na superadequação dos dados de treinamento

`max_iter` - Número máximo de iterações. Padrão é `-1`.

`probability` - Especifica se deve ser habilitado o cálculo de probabilidades. Padrão é `False`.

`random_state` - Semente aleatória. Padrão é `None`.

`shrinking` - Especifica se deve ser habilitado o uso de heurística de encolhimento. Padrão é `True`.

`tol` - Tolerância para critério de parada. Padrão é `1e-3`.

`verbose` - Define o nível de verbosidade. Padrão é `False`.

#### Após a adequação

`support_` - Índices dos vetores de suporte.

`support_vectors_` - Vetores de suporte.

`n_support_` - Número de vetores de suporte para cada classe.

`coef_` - Coeficientes da função de decisão.

### K vizinhos mais próximos

O algortimo KNN (K-Nearest Neighbor, ou K Vizinhos Mais Pŕoximos) faz a classificação com base na distância até algumas amostras (k) de treinamento. A familia de algoritmos é chamada de aprendizado baseado em instâncias (instance-based learning), pois não há parâmetros para serem aprendidos. O modelo tem como base de que apenas a distância é suficiente para fazer uma inferência.

A parte complicada desse tipo de algoritmo está na seleção de atributos, principalmente o atributo `k` que é o número de vizinhos mais próximos.

#### Eficiência na execução

O treinamento possuí complexidade de $O(1)$, mas precisa armazenar dados. Testes $O(Nd)$, em que $N$ é o número de exemplos de treinamento e $d$ é a dimensionalidade.

#### Pré-processamento dos dados

Cálculos baseados em distância tem melhor desempenho com dados padronizados.

#### Para evitar superadequação

Eleve o valor de `k` e mude `p` para a métrica `l1` ou `l2`.

#### Interpretação dos resultados

Interpreta os k vizinhos mais próximos para a amostra (método `.kneighboors`). Esses vizinhos explicam a classificação.

#### Atributos

`algorithm` Pode ser `brute`, `ball_tree` ou `kd_tree`. O padrão é `auto`. `auto` escolhe o algoritmo mais apropriado para o conjunto de dados.

`leaf_size` Tamanho da folha da árvore. Padrão é `30`.

`metric` Métrica de distância, padrão é `minkowski`. Pode ser `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `wminkowski`, `seuclidean`, `mahalanobis`. Além disso aceita um callable, ou seja definido pelo usuário.

`metric_params` Dicionário adicional de parâmetros para a métrica.

`n_jobs` Número de trabalhos em paralelo. Padrão é `None`.

`n_neighbors` Número de vizinhos. Padrão é `5`.

`p` Parâmetro de potência para a métrica `minkowski`. Padrão é `2`.

`weights` Pode ser `uniform` ou `distance`. Padrão é `uniform`. `uniform` atribui pesos iguais para todos os vizinhos. `distance` atribui pesos de acordo com a inversa da distância.

#### Atenção:

Se `k` for um número par e os vizinhos forem separados, o modelo irá retornar uma classe aleatória. Para evitar esse problema, utilize um número ímpar para `k`.

### Árvores de Decisão

Uma árvore de decisão é como ir a um médico que faz uma série de perguntas a fim de determinar a causa de seus sintomas. Pode-se usar um processo para criar uma árvore de decisão e ter uma série de perguntas para prever uma classe alvo. A vantagem é que ele possuí suporte em certos casos para dados não númericos, além disso é necessário pouca preparação dos dados (não há necessidade de escalar), possuí suporte para relacionamentos não lineares. A importância dos atributos é revelada e é de facil entendimento.

O algoritmo padrão usado se chama CART (Classification and Regression Tree, Árvore de Classificação e Regressão). Ele usa a impureza de Gini ou medida de índices para toma de decisões. Isso é feito percorrendo os atributos em um laço e encontrando o valor que forneça a menor probabilidade de erro.

#### Atenção:

Os valores default tendem a resultar em uma superadequação. Utilize método como `max_depth` e validação cruzada para controlar.

#### Eficiência na execução

Para a criação, percorre cada um dos `m` atributos e ordena todas as n amostras, $O(mn log n)$. Para a predição percorre a árvore $O(altura)$.

#### Pré-processamento dos dados

Não tem necessiade do escalonamento. É preciso apenas lidar com os valores ausentes e transformar os dados categóricos em numéricos.

#### Para evitar superadequação

Controle `max_depth` com um numero menor e aumente `min_impurity_decrease`

#### Interpretação dos resultados

É possível percorrer a árvore de opções. Por haver passos, uma árvore não é adequada para relacionamentos lineares (mudanças pequenas tem grandes impactos). A árvore também é extremamente dependente dos dados de treinamento.

#### Atributos

`class_weight` Pesos das classes em um dicionário. `Balanced` definirá valores na proporção inversa das frequências das classes. O default é 1 para cada classe. O padrão é `none`.

`criterion` Função de separação, pode ser `gini` ou `entropy`. O padrão é `gini`.

`max_features` Numero de atributos a serem analisados para separação. Padrão é `None`.

`max_leaf_nodes` Limita o número de folhas, o padrão é `None`.

`min_impurity_decrease` Um nó será dividido se essa divisão diminuir a impureza em pelo menos esse valor. Padrão é `0.0`.

`min_samples_leaf` Número mínimo de amostras em um nó folha. Padrão é `1`.

`min_samples_split` Número mínimo de amostras para dividir um nó. Padrão é `2`.

`min_weight_fraction_leaf` Fração mínima de amostras em um nó folha. Padrão é `0.0`.

`presort` Especifica se deve ser feita uma pré-ordenação dos dados para acelerar o treinamento. Padrão é `False`.

`random_state` Semente aleatória. Padrão é `None`.

`splitter` Estratégia de divisão. Pode ser `best` ou `random`. Padrão é `best`.

#### Atributos após a adequação

`classes_` Classes de destino.

`feature_importances_` Importância dos atributos.

`n_classes_` Número de classes.

`n_features_` Número de atributos.

`tree_`

A visualização da arvore pode ser feita utilizando

```python

import pydotplus
from sklearn.tree import export_graphviz
from io import StringIO

dot_data = StringIO()

export_graphviz(
    tree,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
```


![Exemplo](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/titanic.png)

O pacote `dtreeviz` é útil para compreender as árvores de decisão. ELe cria histogramas contendo rótulos, facilitando a compreensão.

No jupyter é possível exibir um objeto `viz` diretamente

Utilizando o `dtreeviz` é possível visualizar a importância dos atributos.



A importância dos atributos pode ser feita utilizando o `feature_importances_` do modelo.

Ou utilizar o `yellowbrick.features` para visualizar a importância dos atributos. (FeatureImportances)

### Florestas Aleatórias

Uma floresta aleatória (Random Forest) é um conjunto de árvores de decisão. Faz uso de bagging para corrigir a tendência das árvores de decisão à superadequação.

Por criar várias árvores treinadas com subamostras e atributos aleatório a variância é reduzida.

Como o treinmaneto é feito em subamostras dos dados, as florestas aleatoriás são capazes de avaliar o erro OOB e o desempenho. Podem também exibir a importância dos atributos

A intuição de compreender baging vem de um artigo de 1785 do Marquês de Condorcet. Essenialmente, se você tiver um grupo de pessoas que são boas em prever o resultado de um evento, a média de suas previsões será melhor do que a previsão de qualquer indivíduo. Um resumo mais aprofundado pode ser encontrado [aqui](https://en.wikipedia.org/wiki/Wisdom_of_the_crowd). Além disso sempre que uma nova pessoa é adicionada ao grupo, a precisão aumenta.

A ideia das florestas aleatórias é criar uma "floresta" de árvores de decisão treinadas em diferentes colunas dos dados de treinamento. Caso a árvore tenha uma chance melhor que 50% de fazer uma classificação correta, ela deve ser incorporada a predição.

Ela é uma ótima forte para classificação e regressão, entretanto vem perdendo espaço para a gradient boosting

#### Eficiência na execução

Deve criar `j` árvores aleatórias. Isso pode ser feito em paralelo usando `n_jobs`. A complexidade de cada árvore é de $O(mn log n)$, em que cada $n$ é o numero de mostras e `m` o numero de atributos. Para a criação percorre cada um dos `m` atributos em um laço e ordena todas as `n` amostras, $O(mn log n)$. Para a predição percorre a árvore $O(altura)$.

#### Pré-processamento dos dados

Não é necessário.

#### Para evitar superadequação

Adicione mais árvores `n_estimators` e use um valor menor para `max_depth`

#### Interpretação dos resultados

Tem suporte para importância de atributos, porém não há uma única árvore de decisão para percorrer.

#### Parâmetros da instância

`bootstrap` - Especifica se deve ser feita a amostragem com reposição. Padrão é `True`.

`class_weight` - Define o peso das classes. Padrão é `None`. Use dictionary para definir pesos diferentes para cada classe.

`criterion` - Função de separação, pode ser `gini` ou `entropy`. O padrão é `gini`.

`max_depth` - Profundidade máxima da árvore. Padrão é `None`.

`max_features` - Numero de atributos a serem analisados para separação. Padrão é `auto`.

`max_leaf_nodes` - Limita o número de folhas, o padrão é `None`.

`min_impurity_decrease` - Um nó será dividido se essa divisão diminuir a impureza em pelo menos esse valor. Padrão é `0.0`.

`min_samples_leaf` - Número mínimo de amostras em um nó folha. Padrão é `1`.

`min_samples_split` - Número mínimo de amostras para dividir um nó. Padrão é `2`.

`min_weight_fraction_leaf` - Fração mínima de amostras em um nó folha. Padrão é `0.0`.

`n_estimators` - Número de árvores. Padrão é `10`.

`n_jobs` - Número de trabalhos em paralelo. Padrão é `None`.

`oob_score` - Especifica se deve ser habilitado o cálculo do erro OOB. Padrão é `False`.

`random_state` - Semente aleatória. Padrão é `None`.

`verbose` - Define o nível de verbosidade. Padrão é `0`.

`warm_start` - Quando definido como `True`, reutiliza a solução da chamada anterior para ajustar como inicialização. Padrão é `False`.

#### Atributos após a adequação

`classes_` - Classes de destino.

`feature_importances_` - Importância dos atributos.

`n_classes_` - Número de classes.

`n_features_` - Número de atributos.

`0ob_score_` - Erro OOB.

É possível utilizar a importância de Gini para visualizar a importância dos atributos.

O classificador de floresta aleatória calcula a importância dos atributos determinando a diminuição média da impureza para cada atributo (ou seja importância de Gini). A importância de Gini é calculada para cada nó da árvore e ponderada pela probabilidade de atingir esse nó. A importância de Gini é calculada para cada árvore e depois é calculada a média.

Esses valores poderão se tornar imprecisos se os atribtos variarem em escala ou na cardinalidade das colunas de categorias. Uma pontuação mais confiavel é a importância da permutação. Um método mais confiavel é a importância da coluna descartada, entretanto para isso é exigido um novo modelo para cada coluna descartada

### XGboost

O `sklearn` possuí um `GradientBoostedClassifier`, entretanto uma implementação de terceiros consegue ter resultados mais favoraveis.

O XGBoost é uma biblioteca popular, ele cria uma árovre fraca inicialmente e então melhora as àrvores subsequentes. A melhora é feito reduzindo os erros residuais através do `boosting`.

O algoritmo visa capturar qualquer padrão nos erros, até os que parecem aleatoriedade.

#### Eficiência na execução

XGBoost consegue executar em paralelo. Utilizando `n_jobs`, é possivel controlar o numero de CPUs. Caso tenha interesse em um desempenho maior use GPU.

#### Pré-processamento dos dados

Não é necessário o escalonamento com modelos baseados em árvores, entretanto é necessário a codificação de variáveis categóricas.

#### Para evitar superadequação

O parâmetro `early_stopping_rounds=N`, pode ser utilizado para interromper o treinamento caso não haja melhoras após N rodadas. As regularizações L1 e L2 são controladas por `reg_alpha` e `reg_lambda`.

#### Interpretação dos resultados

Exibe a importância dos atributos.

---

O XGBoost tem um parâmetro extra no modelo `fit`. O parâmetro `early_stopping_rounds` pode ser combinado com o parâmetro `eval_set` para interromper o treinamento caso não haja melhoras após N rodadas. `eval_metric` pode ser utilizado para definir a métrica de avaliação.

#### Parâmetros da instância

`max_depth` - Profundidade máxima da árvore. Padrão é `3`.

`learning_rate` - Taxa de aprendizado. Padrão é `0.1`. Também conhecida como ETA. Após cada passo de boosting, os pesos recém adicionados são escalados de acordo com essa taxa. Um valor menor resulta em um modelo mais conservador, mas gasta mais árvores para convergir. Na chamada `.train`, é possível passar parâmetro `learning_rates` para especificar a taxa de aprendizado para cada árvore.

`n_estimators` - Número de árvores. Padrão é `100`.

`silent` - Especifica se deve ser habilitado o modo silencioso. Padrão é `True`.

`objective` - Especifica o objetivo de treinamento. Padrão é `reg:linear`. Pode ser `reg:linear`, `reg:logistic`, `binary:logistic`, `binary:logitraw`, `count:poisson`, `multi:softmax`, `multi:softprob`, `rank:pairwise`, `reg:gamma`, `reg:tweedie`.

`booster` - Especifica o tipo de modelo. Padrão é `gbtree`. Pode ser `gbtree`, `gblinear`, `dart`.

`n_jobs` - Número de trabalhos em paralelo. Padrão é `1`.

`gamma` - Controla a prunning (poda). Varia de 0 a infinito. Padrão é `0`. Essa é a redução de perda mínima necessária para separar mais uma folha. Quanto maior o valor de gama, mais conservador o algoritmo. Caso o a pontuação de treino e teste divergirem, aumente o valor de gama. Caso a pontuação de treino e teste estiverem semelhantes, diminua o valor de gama.

`min_child_weight` - Controla a prunning (poda). Padrão é `1`. Essa é a soma mínima de pesos necessária para separar mais uma folha. Quanto maior o valor, mais conservador o algoritmo.

`max_delta_step` - Deixa as atualizações mais conservadoras. Padrão é `0`.

`subsample` - Fração das amostas a serem utilizadas. Padrão é `1`. Um valor menor resulta em um modelo mais conservador.

`colsample_bytree` - Fração dos atributos a serem utilizados, por rodada. Padrão é `1`.

`colsample_bylevel` - Fração dos atributos a serem utilizados, por nível. Padrão é `1`.


`colsample_bynode` - Fração dos atributos a serem utilizados, por nó. Padrão é `1`.

`reg_alpha` - Regularização L1. Padrão é `0`. Aumente para um modelo mais conservador.

`reg_lambda` - Regularização L2. Padrão é `1`. Aumente para um modelo mais conservador. Essa regularização incentiva pesos menores.

`scale_pos_weight` - Razão entre as classes. Padrão é `1`.

`base_score` - Especifica o valor inicial de todas as previsões. Padrão é `0.5`.

`random_state` - Semente aleatória. Padrão é `0`.

`missing` - Especifica o valor ausente. Padrão é `np.nan`.

`importance_type` - Especifica o tipo de importância. Padrão é `gain`. Pode ser `gain`, `weight`, `cover`, `total_gain`, `total_cover`.

#### Atributos

`coef_` - Coeficientes da função de decisão.

`feature_importances_` - Importância dos atributos.

A importância dos atributos é o ganho médio em todos os nós em que o atributo é usado

O XGBoost é capaz de gerar um gráfico da importância dos atributos. Ele tem um parâmetro `importance_type` cujo valor default é `weight`, entretanto é psosível ajustar para o ganho médio (`gain`) ou `cover`. É bem simples de ser utilizado, basta usar `xgb.plot_importance(model)`. Uma outra opção seria atrvés do yellowbrick com `FeatureImportances`.

Além disso o XGBoost possuí uma representação tanto textual como gráfica das árvores.

O pacote `xgbfir` é uma biblioteca desenvolvida com base no XGBoost, essa biblioteca oferta diversas medidas relacionadas à importância dos atributos.

As medidas fornecidas são:

- `Gain` - O ganho médio em todos os nós em que o atributo é usado.

- `FScore` - O número de vezes que o atributo é usado para dividir os dados em todos os nós.

`wFScore` - Quantidade de possíveis separações em um atributo ou interação entre atributos.

`Avarege wFScore` - `wFScore` dividido pelo número de possíveis separações (FScore).

`Average Gain` - `Gain` dividido pelo número de possíveis separações (FScore).

`Expected Gain` - `Gain` multiplicado pela probabilidade de um nó ser alcançado.

A interface consiste em exportar dados para uma planilha, portanto é recomendavel usar o `pandas`.

![Tabela gerada pelo Xgbfir](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/xgbfir%20output.png)

A partir dessa tabela, pode-se verificar que `sex_male` tem uma posição elevada quanto ao ganho. Enquanto `fare` se destaca pela `WFScore`.

![Interação entre as variáveis](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/interação%20com%20as%20variaveis%20na%20profundidade%201.png)

Com essa tabela é possível verificar a interação entre as variáveis. As duas principais interações ocorrem com `sex_male` e as outras duas variáveis `pclass` e `age`. Esses com certeza são os atributos mais importantes.



### Gradient Boosting com LightGBM

O LightGBM é uma implementação da Microsoft. Ele utiliza um método de amostragem para lidar com valores contínuos. Com isso a criação das árvores é rápida, e o tempo de memória e bem menor.

O LighhGBM também cria árvores em profundidade antes. Com isso para evitar a superadequação utilize `num_leaves`.

#### Eficiência na execução

Consegue utilizar várias CPUs, caso utilize binning pode ser até 15x mais rápido que o XGBoost.

#### Pré-processamento dos dados

Possuí suporte para cofificação de colunas de categorias. AUC costuma ser pior em comparação com codificação one-hot.

#### Para evitar superadequação

Utilize `num_leaves` para controlar a profundidade das árvores. Utilize `min_data_in_leaf` para controlar o número mínimo de amostras em um nó folha. Utilize `max_bin` para controlar o número de bins. Utilize `main_gain_to_split` para controlar a quantidade de ganho necessária para dividir um nó.

#### Interpretação dos resultados

Está disponível, mas costuma ser difíceis de interpretar devido a individualidade fraca das árvores.

#### Parâmetros da instância

`boosting_type` - Especifica o tipo de modelo. Padrão é `gbdt`. Pode ser `gbdt`, `dart`, `goss`.

`class_weight` - Define o peso das classes. Padrão é `None`. Use dictionary para definir pesos diferentes para cada classe. Para problemas de várias classes utilize dicionário, mas para problemas binários utilize `is_unbalance` ou `scale_pos_weight`.

`colsample_bytree` - Fração dos atributos a serem utilizados, por árvore. Padrão é `1`. Isso seleciona uma porcentagem de atributos para cada árvore.

`importance_type` - Especifica o tipo de importância. Padrão é `split`. Pode ser `split`, `gain`. `split` é o número de vezes que o atributo é usado para dividir os dados em todos os nós. `gain` é o ganho médio em todos os nós em que o atributo é usado.

`learning_rate` - Intervalo de (0,10). Esse parâmetro indica a taxa de aprendizado para boosting. Um valor menor pode atrasar a superadequação devido as rodadas possuírem menos 'impacto', mas um numero menor resulta em um desempenho melhor o que pode aumentar o numero de iterações. O padrão é `0.1`.

`max_depth` - Profundidade máxima da árvore. Padrão é `-1`. O valor padrão é -1, o que significa que não há limite de profundidade. Um valor maior resulta em um modelo mais complexo e mais propenso a superadequação.

`min_child_samples` - Número mínimo de amostras em um nó folha. Padrão é `20`. Um valor maior resulta em um modelo mais conservador. Numeros menores podem resultar em superadequação.

`min_child_weight` - Controla a prunning (poda). Padrão é `1`. Essa é a soma mínima de pesos necessária para separar mais uma folha. Quanto maior o valor, mais conservador o algoritmo.

`min_split_gain` - Redução mínima de perda necessária para dividir um nó. Padrão é `0`. Um valor maior resulta em um modelo mais conservador.

`n_estimators` - Número de árvores. Padrão é `100`.

`n_jobs` - Número de trabalhos em paralelo. Padrão é `1`.

`num_leaves` - Número de folhas. Padrão é `31`. Um valor maior resulta em um modelo mais complexo e mais propenso a superadequação.


`objective` - Especifica o objetivo de treinamento. Padrão é `regression`. Pode ser `regression`, `regression_l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`, `binary`, `multiclass`, `multiclassova`, `cross_entropy`, `cross_entropy_lambda`, `lambdarank`.

`random_state` - Semente aleatória. Padrão é `None`.

`reg_alpha` - Regularização L1. Padrão é `0`. Aumente para um modelo mais conservador.

`reg__lambda` - Regularização L2. Padrão é `0`. Aumente para um modelo mais conservador. Essa regularização incentiva pesos menores.

`silent` - Especifica se deve ser habilitado o modo silencioso. Padrão é `True`.

`subsample` - Fração das amostas a serem utilizadas. Padrão é `1`. Um valor menor resulta em um modelo mais conservador.

`subsample_for_bin` - Número de amostras para construir bins. Padrão é `200000`.

`subsample_freq` - Frequência de amostragem. Padrão é `0`.

---

A importância com o numero de atributos (numero de vezes utilizado) pode ser feita utilizando (`.feature_importances_`).

A biblioteca `LightGBM` é capaz de gerar um gráfico da importância dos atributos com o método `plot_importance`. É possível utilizar o parâmetro `importance_type` para especificar o tipo de importância. O padrão é `split`, mas pode ser `gain`.

Também é possível gerar uma árvore de decisões com o método `plot_tree`. É possível utilizar o parâmetro `num_tree` para especificar o número da árvore. O padrão é `0`.

### TPOT 

O TPOT usa um algoritmo genético para testar diferentes modelos e ensembles (conjuntos). Pode demorar muito tempo para executar, pois o algoritmo considera diversos modelos e passos de pré-processamento, além de testar os hiperparâmetros.

#### Eficiência na execução

O TPOT pode demorar muito tempo para executar, pois o algoritmo considera diversos modelos e passos de pré-processamento, além de testar os hiperparâmetros. Utilize `n_jobs` para controlar o número de CPUs. Para usar todas use `-1`.

#### Pré-processamento dos dados

Deve ser feita a remoção de `NaN` e a codificação de variáveis categóricas.

#### Para evitar superadequação

O ideal é utilizar validação cruzada para minimizar a superadequação.

#### Interpretação dos resultados

Depende dos resultados. O TPOT pode ser utilizado para encontrar o melhor modelo e depois utilizar esse modelo para interpretar os resultados.

#### Parâmetros da instância

`generations` - Número de gerações. Padrão é `100`.

`population_size` - Tamanho da população para o algoritmo genético. Padrão é `100`. Quanto maior o valor, mais tempo leva para executar.

`offspring_size` - Define o número de descendentes para cada geração. Padrão é `100`.

`mutation_rate` - Define a taxa de mutação. Padrão é `0.9`.

`crossover_rate` - Define a taxa de cruzamento(quantidade de pipelines criados em cada geração). Padrão é `0.1`.

`scoring` - Especifica a métrica de avaliação. Padrão é `accuracy`. Pode ser `accuracy`, `adjusted_rand_score`, `average_precision`, `balanced_accuracy`, `f1`, `f1_macro`, `f1_micro`, `f1_samples`, `f1_weighted`, `neg_log_loss`, `neg_mean_absolute_error`, `neg_mean_squared_error`, `neg_mean_squared_log_error`, `neg_median_absolute_error`, `precision`, `precision_macro`, `precision_micro`, `precision_samples`, `precision_weighted`, `r2`, `recall`, `recall_macro`, `recall_micro`, `recall_samples`, `recall_weighted`, `roc_auc`.

`cv` - Especifica a estratégia de validação cruzada. Padrão é `5`. Pode ser `None`, `kfold`, `stratifiedkfold`, `groupkfold`, `timeseriesplit`, `shuffle`, `stratifiedshuffle`, `groupshuffle`, `timeseriessplit`.

`subsample` - Fração das amostas a serem utilizadas. Padrão é `1`. Um valor menor resulta em um modelo mais conservador.

`max_time_mins` - Tempo máximo em minutos. Padrão é `None`.

`max_eval_time_mins` - Tempo máximo em minutos para avaliação de um pipeline. Padrão é `5`.

`random_state` - Semente aleatória. Padrão é `None`.

`config_dict` - Dicionário de configuração. Padrão é `None`.

`warm_start` - Quando definido como `True`, reutiliza a solução da chamada anterior para ajustar como inicialização. Padrão é `False`. As chamadas dizem respeito ao método `fit`.

`memory` - Especifica se deve ser habilitado o uso de memória. Padrão é `None`.

`use_dask` - Especifica se deve ser habilitado o uso de dask. Padrão é `False`.

`periodic_checkpoint_folder` - Especifica o diretório para salvar os checkpoints. Padrão é `None`.

`early_stop` - Especifica se deve ser habilitado o uso de early stop. Padrão é `None`.

`verbosity` - Define o nível de verbosidade. Padrão é `0`.

`disable_update_check` - Especifica se deve ser habilitado o uso de atualização. Padrão é `False`.

#### Atributos

`evaluated_individuals_` - Indivíduos avaliados.

`fitted_pipeline_` - Pipeline ajustado. Com o seu uso é possivel exportar o pipeline com o método `export`.

### Códigos

Os códigos deste capítulo podem ser encontrados [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo10.ipynb).

## Capítulo 11 - Seleção do Modelo

### Curva de Validação

Uma curva de validação é uma representação visual de determinar um valor apropriado para um hiperparâmetro. Essa curva é um gráfico que mostra como o desenpenho do modelo muda com o valor do hiperparâmetro.

![Validation Curve](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/validation%20curve.png)

O gráfico mostra tanto os dados de treinamento como de validação. Dessa forma é possivel estimar como o modelo reage a diferentes valores de hiperparâmetros. No geral a escolha e sempre visando a maxima pontuação dos dados de validação

O gráfico acima foi gerado utilizado o `Yellowbrick`, o foco foi na verificação do hiperparâmetro `max_depth`, além disso é possível fornecer um parÂmetro `scoring` para especificar a métrica de avaliação.

### Curva de Aprendizagem

Uma curva de aprendizagem é uma representação visual de determinar se um modelo está sofrendo de superadequação ou subadequação. Essa curva é um gráfico que mostra como o desenpenho do modelo muda com o tamanho do conjunto de treinamento.

O gráfico mostra as instâncias de treinamento e a pontuação para validação cruzada à medida que cria-se modelos com mais amostras. 

Caso a pontuação da validação cruzada tenda a ser incrementada, é um sinal de que mais dados podem ajudar.

![Learning Curve](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/learning%20curve.png)

Se houver variailidade (uma área sombreada grande) na pontuação dos dados, significa que o modelo sofre de erros de bias e é simples demais (subadequação).

Se houver variabilidade na pontuação para validação cruzada, é porque o modelo está sujeito a erros de variância e é muito complexo (superadequação).

Um outro fator que pode mostrar uma superadequação está no fato de que o desempenho para conjunto de dados de validação é muito pior do que o desempenho para o conjunto de dados de treinamento.

### Códigos

Os códigos deste capítulo podem ser encontrados [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo11.ipynb).

## Capítulo 12 - Métricas e Avaliação de Classificação

### Matriz de Confusão

Uma matriz de confusão é extremamente relevante para auxiliar na compreensão do desempenho de um classificador. Um classificador binário pode ter quatro resultados de classificação:

- Verdadeiro Positivo (TP) - O modelo previu corretamente a classe positiva.

- Verdadeiro Negativo (TN) - O modelo previu corretamente a classe negativa. 

- Falso Positivo (FP) - O modelo previu incorretamente a classe positiva. Em outras palavras, o modelo previu que era positivo quando na verdade era negativo.

- Falso Negativo (FN) - O modelo previu incorretamente a classe negativa. Em outras palavras, o modelo previu que era negativo quando na verdade era positivo.

Supondo que positivo é ter câncer e negativo é não ter câncer, um falso negativo seria um paciente que tem câncer, mas o modelo previu que não tinha. Um falso positivo seria um paciente que não tem câncer, mas o modelo previu que tinha. Esses erros são os mais comuns, chamados de erro do tipo I e erro do tipo II.

|Real | Previsto como negativo | Previsto como positivo |
|-----|------------------------|------------------------|
|Negativo | Verdadeiro Negativo | Falso Positivo tipo I|
|Positivo | Falso Negativo tipo II | Verdadeiro Positivo|

```python

tp=( # true postive
    (y_test == 1) & (y_test == y_pred)
).sum()

tn=( # true negative
    (y_test == 0) & (y_test == y_pred)
).sum()

fp=( false positive
    (y_test == 0) & (y_test != y_pred)
).sum()

fn=( # false negative
    (y_test == 1) & (y_test != y_pred)
).sum()

```

Usando `scikit-learn`, é possível utilizar a função `confusion_matrix` para gerar a matriz de confusão.

Usando o `Yellowbrick`, é possível utilizar a função `ConfusionMatrix` para gerar a matriz de confusão. 

![Confusion Matrix](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/confusion%20matrix.png)

### Métricas

O `scikit-learn` possuí diversas métricas muito comuns para classificação como:

- `accuracy:` A acurácia é a proporção de previsões corretas. É a métrica mais comum para classificação binária. A acurácia é calculada como a proporção de previsões corretas em relação ao número total de previsões. A acurácia é uma métrica útil quando as classes estão bem equilibradas.

- `average_precision:` A precisão média é a área sob a curva de precisão-recall. A precisão média é uma métrica útil quando as classes estão desequilibradas. É um resumo da curva de precisão e recall.

- `f1:` O F1 é a média harmônica da precisão e do recall. O F1 é uma métrica útil quando as classes estão desequilibradas. É uma métrica útil quando as classes estão desequilibradas.

- `neg_log_loss:` A perda logarítmica negativa é uma métrica de perda. A perda logarítmica negativa é útil quando as classes estão desequilibradas.

- `precision:` A precisão é a proporção de previsões corretas positivas. 


- `recall:` O recall é a proporção de previsões corretas positivas. 

- `roc_auc:` A área sob a curva ROC. 

Essas strings, pdem ser usadas no parâmetro scoring em uma busca de grade, por exemplo.

#### Acurácia

Porcentagem de classificações corretas:

$$
\frac{TP + TN}{TP + TN + FP + FN}
$$

Nem sempre um modelo com alta acurácia é o melhor. Por exemplo, se o modelo prevê que todos os pacientes não têm câncer, a acurácia será alta, mas o modelo não é útil.

Conforme mencionado o `scikit-learn` possuí a função `accuracy_score` para calcular a acurácia.

#### Recall

Porcentagem de positivos corretamente classificados:

$$
\frac{TP}{TP + FN}
$$

Possuí implementação no `scikit-learn` com a função `recall_score`.

#### Precisão

Porcentagem de positivos corretamente classificados:

$$
\frac{TP}{TP + FP}
$$

Possuí implementação no `scikit-learn` com a função `precision_score`.

#### F1

Média harmônica entre precisão e recall:

$$
pre = \frac{TP}{TP + FP}
$$

$$
rec = \frac{TP}{TP + FN}
$$

$$
F1 = 2 \times \frac{pre \times rec}{pre + rec}
$$

Possuí implementação no `scikit-learn` com a função `f1_score`.

### Relatório de Classificação

O `Yellowbrick` possuí uma função para gerar um relatório de classificação. O relatório de classificação é uma tabela que mostra as métricas de classificação para cada classe. O relatório de classificação é útil quando as classes estão desequilibradas.

O relatório é colorido e quanto mais próximo do vermelho(mais próximo de um), melhor será.

![Classification Report](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/classification%20report.png)

### ROC (Receiver Operating Characteristic)

A curva ROC mostra o desempenho do classificador, exibindo a taxa de verdadeiros positivos (recall/sensibilidade) à medida que a taxa de falsos-positivos (especificidade inversa) é variada. A curva ROC é extremamente útil

Uma regra geral é que o gráfico deve ter uma protuberância em direção ao canto superior esquerdo. Quanto mais próximo o gráfico estiver do canto superior esquerdo, melhor será o desempenho do classificador.

Usando o `sklearn` é possível obter o valor com `sklearn.metrics import roc_auc_score`, já com o `Yellowbrick ` é possível utilizar a função `ROCAUC` para gerar o gráfico.

![ROC](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/ROC.png)

### Curva de precisão-recall

A curva ROC pode ser enganosa quando as classes estão desequilibradas. A curva de precisão-recall é uma alternativa útil quando as classes estão desequilibradas.

Uma classificação é uma tarefa de balanceamento de modo a encontrar tudo de que é necessário (recall), ao mesmo tempo que limita os resultados ruins (precisão). A curva de precisão-recall mostra a precisão e o recall para diferentes limiares de probabilidade.

Utilizando `sklearn.metrics import average_precision_score` é possível obter o valor da área sob a curva. Já com o `Yellowbrick` é possível utilizar a função `PrecisionRecallCurve` para gerar o gráfico.

![Curva precisão recall](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/Curva%20precisão%20recall.png)

### Gráfico de Elevação

O gráfico de elevação (lift curve) é outra forma de olhar a informação que está em um gráfico de ganhos cumulativos. A elevação, demonstra o quão melhor é o nosso desempenho em relação ao modelo de base. O modelo de base é aquele que não faz nenhuma previsão, ele simplesmente prevê a classe mais frequente. A própria biblioteca do `scikitplot` tem uma função para gerar o gráfico de elevação.

`scikitplot.metrics.plot_lift_curve(y_test, y_probas)`

![Gráfico de Elevação](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/gráfico%20de%20elevação.png)

### Balanceamento das classes

O `Yellowbrick` tem um gŕafico de barras simples para visualizar os tamanhos das classes. Quando os tamanhos relativos das classes são diferentes, o modelo pode ter dificuldade em aprender a classe minoritária. O gráfico de barras é uma maneira rápida de verificar se as classes estão desequilibradas.

Ao realizar a separação dos dados em conjunto de treinamento e de teste, é recomendavél utilizar amostragem estratificada, de modo a manter a proporção das classes.

A própria função `test_train_split` do `sklearn` já possui um parâmetro `stratify` para realizar a amostragem estratificada.

![Balancemaneto das classes](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/balanceamento%20de%20classes.png)

### Erro de predição de classe

O gráfico de erro de predição de classe do `Yellowbrick` é um gráfico em barras para exibir uma matriz de confusão. 

![Erro de predição de classe](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/Erro%20de%20predição%20de%20classe.png)

Na parte superior da barra a esquerda estão as pessoas que morreram, mas para as quais foi previsto que sobreviveram (falsos-positivos). Na parte inferior da barra a direita estão as pessoas que sobreviveram, mas para as quais foi previsto que morreram (falsos-negativos).

### Limiar de Discriminação

A maioria dos classificadores binarios que fazem a predição de probabilidades tem um limiar de discriminação de 50%. Caso a probabilidade prevista esteja acima de 50%, o classificador irá atribuir um rótulo positivo. Caso a probabilidade prevista esteja abaixo de 50%, o classificador irá atribuir um rótulo negativo.

Este gráfico pode ser útil para visualizar a relação de compromisso entre precisão e recall. A taxa de fila é a porcentagem de previsões acima do limiar. Ela pode ser considerada como a porcentagem de casos a serem analisados.

O `Yellowbrick` disponibiliza essa visualização, ele faz o embaralhamento dos dados e executa 50 tentativas por padrão, separando 10% para validação. Use `yellowbrick.classifier import DiscriminationThreshold` para gerar o gráfico.

![Limiar de discriminação](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/limiar%20de%20discriminação.png)

### Códigos

Os códigos deste capítulo podem ser encontrados [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo12.ipynb).

## Capítulo 13 - Explicando os Modelos

### Coeficientes de Regressão

Os interceptos e os coeficientes de regressão explicam o valor esperado e como os atributos causam impacto na predição. Um coeficiente positivo indica que, à medida que o valor de um atributo aumento, o valor esperado também aumenta. Um coeficiente negativo indica que, à medida que o valor de um atributo aumenta, o valor esperado diminui.

Uma explicação mais completa sobre como funciona o método de Least Squares pode ser encontrada [aqui](https://github.com/pcmoraesmenezes/Data-Science-and-Machine-Learning-Course/blob/main/Machine%20Learning%20Crash/practice/model_/linear_regression.ipynb)

### Importância dos Atributos

Modelos baseados em árvore do `scikit-learn`, incluem atributos como `.feature_importances_` para inspecionar como os atributos de um conjunto de dados afetam o modelo. A inspeção pode ser realizada tanto de forma gráfica quanto de forma textual.

### LIME (Local Interpretable Model-Agnostic Explanations)

O LIME, consegue ser de grande ajuda na explicação de modelos caixa-preta. Ele realiza uma interpretação local, em vez de uma interpretação global. O LIME é um método de interpretação de modelo agnóstico, ou seja, pode ser aplicado a qualquer modelo.

Para um determinado ponto de dado ou amostra, o LIME consegue mostrar os atributos importantes para determinar o resultado. O método para ele realizar isto é causando uma perturbação nos dados e observando como o modelo reage. O LIME é capaz de explicar modelos de classificação e regressão.

![LIME](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/LIME.png)

É possível também usar o `matplotlib` para exportar a explicação do `LIME`

Uma coisa interessante sobre esses dados, que como mencionado, o LIME analisa a importância dos atributos, ou seja, pequenas modificações que forem feitas, podem ser de grande efeito no modelo no geral, como por exemplo alterando o sexo dos passageiros, o modelo pode ter uma grande mudança.

### Interpretação de Árvores

Para os modelos baseados em árvore do `sklearn`, é possível usar o pacote `treeinterpreter`. Através desse pacote, é possível calcular o bias(víes) e a contribuição de cada atributo para cada previsão. O bias é a média de todas as previsões. A contribuição de cada atributo é a diferença entre a previsão com o atributo e a previsão sem o atributo.

Cada contribuição lista como o atributo contribui com cada um dos rotulos.

![Interpretação de árvores](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/interpretação%20de%20árvores.png)

`Prediction` - Isso indica que o modelo previu duas classes  e fornece as probabilidades associadas a cada classe. No caso a classe 0 tem uma probabilidade de cerca de 82% enquanto a classe 1 tem a probabilidade de cerca de 18%.

`Bias` - Ele apresenta a média das saídas do conjunto de treinamento

`Feature contributions`: Essa parte mostra a contribuição de cada recurso para a predição, onde cada linha representa um recurso e as contribuições positivas e negativas para cada classe, por exemplo pclass, onde teve uma contribuição de aproximadamente 0.028 para a classe 0 e -0.028 para a classe 1.

### Gráficos de Dependência Parcial

É notavél que os atributos causam um impacto nos modelos, mas não se é sabido como o impacto varia à medida que o valor do atributo muda. Os gráficos de dependência parcial, são extremamente uteis para preencher essa lacuna.

Utilizando o `pdpbox` é possível gerar os gráficos de dependência parcial.

![Dependência parcial](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/dependencia%20parcial.png)

### Modelos Substitutos

No caso de modelos não interpretaveis como uma rede neural ou SVM (Support Vector Machine), é possível utilizar modelos substitutos para interpretar o modelo. O modelo substituto é um modelo simples que é treinado para imitar o modelo complexo. O modelo substituto pode ser interpretado para entender o modelo complexo.

O `sklearn` possuí o `DecisionTreeClassifier` que pode ser utilizado como modelo substituto.

### SHAP (SHapley Additive exPlanations)

Esse pacote é capaz de exibir as contribuições dos atributos para qualquer modelo. É um pacote bem interessante!

Ele funciona tanto para classificação quanto para regressão e gera valores `SHAP`. Para modelos de classificação, o valor SHAP é a soma dos log odds(logaritmo das chances) em uma classificação binaria.Para regressão os valores SHAP são os valores preditos.

![SHAP](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/SHAP.png)

Este é um exemplo de gráfico de forças, para a amostra 20

#### Atenção

O SHAP precisa de iniciar o `js` atraves de `shap.initjs()`, isso permitira a geração de gráficos e o código não ficará pesado

---

![SHAP](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/SHAP%202.png)

Visualização completa das importâncias

![SHAP](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/shap%203.png)

![SHAP](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/shap4.png)

Uma pontuação baixa de sexo masculino impulsiona a sobrevivência por exemplo.

### Códigos

Os códigos deste capítulo podem ser encontrados [aqui](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/codigos/capitulo13.ipynb).

## Capítulo 14 - Regressão

A regressão é um processo de machine learning supervisionado. É bem semelhante ao modelo de classificação, a diferença está em que enquanto em um modelo de classificação a predição é realizada através de um rotulo, em um modelo de regressão a predição é realizada através de um valor numérico.

Ou seja, caso se deseja realizar uma predição númerica, como o valor de uma casa, uma ação no mercado, o modelo de regressão é mais indicado.

O `sklearn` se mostra incrível mais uma vez nesse quiesito pois, é capaz de aplicar muitos dos mesmos modelos de classificação em problemas de regressão. Os metódos são praticamente o mesmo, usa-se `.fit`, `.score` e `.predict`. O mesmo para as bibliotecas de boosting da próxima geração como `XGBoost` e `LigthGBM`.

Outra grande diferença está nas métricas de avaliação, que serão abordadas posteriormente. O conjunto utilizado para este estudo sera o conjunto de dados habitacionais de Boston.

Foi realizado o download deste dataset pois o `sklearn`, devido a questões eticas desconsiderou o uso deste dataset.

O target deste dataset é a variável `MEDV` que representa o valor mediano das casas ocupadas pelos proprietários em milhares de dólares.

Uma breve descrição dos atributos pode ser encontrada no código `capitulo14.ipynb`, ele será disponilizado ao fim da sessão.

### Modelo de base

Um modelo de regressão básico, dará algo com o qual pode ser comparado aos outros modelos. 

O `sklearn`, o resultado default do método `.score` é o coeficiente de determinação $(r^{2})$ ou $(R^{2})$. Esse valor serve para explicar o percentual de variação dos dados de entrada capturado pela predição. No geral, o valor estará entre $0$ e $1$, mas pode ser negativo se o modelo for muito ruim.

A estratégia de `DummyRegressor` é prever o valor médio do conjunto de treinamento. Esse modelo geralmente não tem um bom desempenho, mas é um bom ponto de partida para comparação.

### Regressão Linear

Uma regressão linear é uma técnica relativamente simples que utiliza o método de mínimos quadrados para ajustar uma linha aos dados. O método de mínimos quadrados é um método de otimização que minimiza a soma dos quadrados das diferenças entre os valores reais e os valores previstos.

Eu ja desenvolvi um repositorio explicando sobre o método de minimos quadrados e uma aplicação usando python, pode ser acessada [aqui](https://github.com/pcmoraesmenezes/Calculo/blob/main/Explica%C3%A7%C3%A3o%20de%20Modelos/Regress%C3%A3o%20Linear.md)

Retomando ao modelo de Regressão Linear, ele tenta realizar uma adequação da fórmula da reta $y = mx+b$, ao mesmo tempo que minimiza o quadrado dos erros.

Quando o modelo encontra os valores, tem-se o intercepto e um coeficiente. O intercepto entrega um valor de base para uma predição, modificado pela soma do produto entre o coeficiente e o dado de entrada.

Esse formato é facilmente generalizado para dimensões maiores. Nesse caso, cada atributo terá um coeficiente. Quanto maior o valor absoluto do coeficiente, mais impacto terá o atributo no alvo.

Esse modelo pressupõe que a predição é uma combinação linear dos dados de entrada. Em alguns casos a predição será perfeita, esses casos são os que não encontra-se "ruídos" nso dados, ou seja, os dados são perfeitamente lineares. Por exemplo uma formula da reta $y = 2x + 1$.

Entretanto para alguns dados, isso não será suficiente, e na verdade a complexidade pode ser forçada utilizando a transformação dos atributos `preprocessing`.

Tem-se também a `PolynomialFeatures` do `sklearn`, que é capaz de criar combinações polinomiais dos atributos, pode resultar em uma superadequação em alguns casos. Já debati também sobre esse problema da superadequação, principalmente no caso da regressão linear, pode ser acessado [aqui](https://github.com/pcmoraesmenezes/Data-Science-and-Machine-Learning-Course/blob/main/Machine%20Learning%20Crash/practice/high_capacity_problem/overfitting.ipynb). Nesse exemplo ficou bem claro que quanto maior o `degree` do modelo, melhor ele consegue se ajustar aos dados já presentes, entretanto ele não consegue generalizar para novos dados. Uma saída seria utilizar os regressores `ridge` e `lasso` para regularizar o estimador.

Um dos outros defeitos deste modelo é a ideia de `heterocedasticidade`. A heterocedasticidade nada mais é que: A medida que os valores de entrada mudam, o erro da predição geralmente mudam também. Se colocar os dados de entrada e os resíduos(predição) em um gráfico, é possível ver que os resíduos aumentam à medida que os valores de entrada aumentam. Isso é um problema porque o modelo não consegue prever com precisão os valores de entrada maiores. O gráfico ficará em um formato de leque ou cone.

Um outro fator é a questão da multicolinearidade, na qual se as colunas forem altamente correlacionadas, será mais dificil de interpretar os coeficientes, mas isso não trará impacto no modelo, apenas nos coeficientes.

#### Eficiência na execução

Utilize `n_jobs` para controlar o número de CPUs. Para usar todas use `-1`.

#### Pré processamento dos dados

É necessário uma padronização dos dados antes do treinamento.

#### Para evitar superadequação

Simplificação no modelo costuma ser a forma mais eficiente, mas no geral evitando o uso de atributos polinomiais costuma ser efetivo.

#### Interpretação dos resultados

É possível interpretar os resultados como pesos para contribuição dos atributos, mas é feito uma suposição de que os atributos tenham uma distribuição normal e sejam independentes, ou seja mais uma vez a questão da multicolinearidade.

É possível remover atributos colineares para facilitar a interpretação.

$R^2$ informará até que ponto a variância total do resultado é explicada pelo modelo.

#### Parâmetros da instância

- `n_jobs = None` - Número de trabalhos em paralelo. Padrão é `None`.

#### Atributos após a adequação

- `coef_` - Coeficientes para cada atributo.

- `intercept_` - Intercepto.

O valor de `.intercept_` é o valor médio esperado. Aqui é onde tem-se o maior peso do escalonamento dos dados, pois o escalonamento dos dados afeta os coeficientes. O sinal destes, explica a direção da relação entre o atributo e o target. Um sinal positivo, indica que o atributo e o target aumentam juntos. Um sinal negativo, indica que o atributo e o target diminuem juntos. Ou seja são proporcionais.

Os coeficientes podem ser vistos utilizando `Yellowbrick`.

![Imagem](/Livros/Machine%20Learning%20-%20Guia%20de%20Referência%20Rápida/images/atributos_linear.png)

Esse é o gráfico da importância dos atributos. Pode ser visto que RM, aumenta o preço, a idade não impacta muito, mas o LSTAT, diminui o preço.

### SVMs

As SVMs (Support Vector Machines) também podem ser utilizadas para fazer regressão.

Elas possuem as seguintes propriedades:

#### Eficiência na execução

A implementação do `scikit-learn` é $O(n^{4})$, ou seja a medida que a quantidade de dados aumenta, ela não terá um bom desempenho. Há opções para lidar com esse problema, como utilizar um kernel linear ou o modelo `LinearSVR`, dessa forma é possível melhorar o desempenho da execução, claro que talvez a acurácia sofra um pouco. Uma outra saída seria utilizar o `cache_size` para reduzir a ordem para $O(n^{3})$.

#### Pré processamento dos dados

O algoritmo não é invariante a escala, portanto é necessário realizar a padronização dos dados antes do treinamento.

#### Para evitar superadequação

É possível utilizar o parâmetro `C` para controlar a complexidade do modelo. Quanto maior o valor de `C`, mais complexo será o modelo. Um valor menor permite uma margem menor no huperplano. Um valor maior para `gamma` tenderá a uma superadequação nos dados de treinamento. O modelo de `LinearSVR` aceita parâmetros `loss` e `penalty` para controlar a complexidade. Além disso o parâmetro `epsilon` pode ser aumentado.

#### Interpretação dos resultados

Realize a inspeção `.support_vectors_`, a interpretação pode ser um pouco confusa. Já para kernels lineares, é possível inspecionar os coeficientes `.coef_` e o intercepto `.intercept_`.

#### Parâmetros da instância

- `C = 1.0` - Parâmetro de regularização. Padrão é `1.0`. Quanto menor o valor, mais estreita sera a fronteira de decisão, podendo aumentar a superadequação

- `cache_size = 200` - Especifica o tamanho do cache em MB. Padrão é `200`. Aumentar esse valor pode melhorar o tempo de treinamento.

- `coef0 = 0.0` - Parâmetro para kernels polinomiais e sigmoid. Padrão é `0.0`.

- `epsilon = 0.1` - Especifica a margem de erro. Padrão é `0.1`.

- `degree = 3` - Grau do polinômio para kernels polinomiais. Padrão é `3`.

- `gamma = auto` - Coeficiente do kernel. Pode ser um numero, `scale` ou `auto`. Padrão é `auto`. Um valormneor resulta em superadequação.

- `kernel = rbf` - Especifica o kernel. Padrão é `rbf`. Pode ser `linear`, `poly`, `rbf`, `sigmoid`, `precomputed`.

- `max_iter = -1` - Número máximo de iterações. Padrão é `-1`, ou seja, sem limite.

- `probability = False` - Especifica se deve ser habilitado o uso de probabilidade. Padrão é `False`.

- `random_state = None` - Semente aleatória. Padrão é `None`.

- `shrinking = True` - Especifica se deve ser habilitado o uso de shrinking. Padrão é `True`.

- `tol = 0.001` - Tolerância para critério de parada. Padrão é `0.001`.

- `verbose = False` - Especifica se deve ser habilitado o uso de verbosidade. Padrão é `False`.

#### Atributos após a adequação

- `coef_` - Coeficientes para cada atributo.

- `intercept_` - Intercepto.

- `support_vectors_` - Vetores de suporte.

- `support_` - Índices dos vetores de suporte.

### K-Nearest Neighbors

KNN(K-Nearest Neighboors, ou K vizinhos mais próximos) também aceita regressão, e encontra k vizinhos alvos para a amostra sobre a qual se deseja fazer uma predição. A predição é a média dos valores alvos dos vizinhos.

#### Eficiência na execução

O tempo de treinamento é $O(1)$, mas os dados das amostras devem ser armazenados. O tempo de teste é $O(Nd)$, em que $N$ é o numero de exemplos de treinamento e $d$ a dimensionalidade.

#### Pré processamento dos dados

Por ser calculo baseado nas distancias entre os dados, estes costumam ter melhores desenvolvimentos quando os dados são padronizados.

#### Para evitar superadequação

Eleve `n_neighbors` para reduzir a complexidade do modelo. Mude `p` para métrica `L1` ou `L2` para controlar a complexidade.

#### Interpretação dos resultados

Utilize o método `.kneighboors`, esses vizinhos podem explicar a predição.

#### Parâmetros da instância

- `algorithm = auto` - Pode ser `brute`, `ball_tree` ou `kd_tree`

- `leaf_size = 30` - Para algoritmos baseados em árvores.

- `metric = minkowski` - Métrica de distância.

- `metric_params = None` - Dicionario adicional de parametros para métrica personalizada.

- `n_jobs = 1` - Numero de CPUs

- `n_neighboors = 5` - Número de vizinhos.

- `p = 2` - Parâmetro de potência de Minkowski: 1 = manhattan (L1); 2 = euclidiana

- `weights = 'uniform'` - Pode ser `distance`, caso em que pontos mais próximos terão mais influencia.

### Árvore de Decisão

As árvores de decisão também aceitam os modelos de regressão. Em cada nível da árvore, várias separações nos atributos são avaliadas. A separação que gerar o menor erro de regressão é escolhida. O erro de regressão é a soma dos quadrados das diferenças entre os valores reais e os valores previstos.

O parâmetro `criterion` pode ser ajustado para estabelecer uma métrica de erro (impureza).

#### Eficiência na execução

Para a criação, é percorrido cada um dos $m$ atributos e ordenadado todas as $n$ amostras. Portanto a complexidade de execução na criação é de $O(m n log n).$

#### Pré processamento dos dados

Não é necessario o escalonamento, entretanto é crucial a imputação dos dados faltantes.

#### Para evitar superadequação

Altere o atributo `max_depth` com um número menor e aumente `min_impurity_decrease`.

#### Interpretação dos resultados

É possível percorrer a árvore de opções. Por haver passos, uma árvore pode ser ruim para lidar com relacionamentos lineares, ou seja uma pequena alteração nos atributos pode gerar uma árvore completamente nova. A árvore também depende muito de dados de treinamento, ou seja, se os dados de treinamento forem alterados, a árvore também será.

#### Parâmetros da instância

- `criterion = mse` - Pode ser `mse` ou `friedman_mse` ou `mae`. Essa seria a função de separação. O padrão é o erro quadrado médio (Mean Squared Error).

- `max_depth = None` - Profundidade máxima da árvore. Padrão é `None`. O default é sempre construir a árvore até que todas as folhas sejam puras ou até que todas as folhas contenham menos que `min_samples_split` amostras.

- `max_features = None` - Número máximo de atributos a serem considerados para a separação. Padrão é `None`.

- `max_leaf_nodes = None` - Número máximo de folhas. Padrão é `None`.

- `min_impurity_decrease = 0.0` - Um nó será dividido se essa divisão diminuir a impureza em pelo menos esse valor. Padrão é `0.0`.

`min_samples_leaf = 1` - Número mínimo de amostras em uma folha. Padrão é `1`.

- `min_samples_split = 2` - Número mínimo de amostras para dividir um nó. Padrão é `2`.

- `min_weight_fraction_leaf = 0.0` - Fração mínima de amostras em uma folha. Padrão é `0.0`. Esse seria o mínimo de pesos para uma folha.

- `presort = False` - Especifica se deve ser habilitado o uso de presort. Padrão é `False`. Isso pode agilizar o treinamento com um conjunto de dados menor ou com uma profundidade restrita se definido como `True`.

- `random_state = None` - Semente aleatória. Padrão é `None`.

- `splitter = best` - Pode ser `best` ou `random`. Padrão é `best`.

#### Atributos após a adequação

- `feature_importances_` - Importância dos atributos. Vem em formato de array de Gini.

- `max_features_` - Número de atributos.

- `n_features_` - Número de atributos.

- `n_outputs_` - Número de saídas.

- `tree_` - Estrutura da árvore.

---

É possível realizar a visualização da árvore com o `pydotplus`.

Limitando a quantidade de nós da árvore tem-se a seguinte representação:

![Árvore de Decisão](/images/arvore2.png)

Além disso com a ajuda do pacote `dtreeviz` é possível visualizar um gráfico de dispersão em cada um dos nós da árvore.

![Árvore de Dispersão](/images/arvore3.svg)

Além disso a importância dos atributos pode ser encontrada utilizando `dtr.feature_importances_`.