# Introdução à Inteligência Artificial: Uma abordagem não técnica - Tom Taulli

Data de publicação do livro -> 17 de Dezembro de 2019

### Sistema Inteligente x Sistema Esperto:

- Um sistema inteligente dispõe de informações e funcionalidades que empoderam o usuário a tomar decisões mais inteligentes.

- Um sistema inteligente é capaz de aprender com o usuário e com o ambiente.

- Um sistema esperto esconde o processo de tomada de decisão do usuário.

- Um sistema esperto é capaz de tomar decisões sem a necessidade de interação com o usuário.

- O usuario então é passivo no sistema esperto (ele simplesmente consome as informações dadas pela máquina), enquanto que no sistema inteligente o usuário é ativo.

## Introdução:

Inteligência artificial está presente em diversos aspectos do nosso dia a dia, por exemplo em sistemas de recomendação de filmes, músicas, livros, etc.
Um dos aplicativos que faz um uso muito bom de IA é o Uber, que utiliza IA para calcular o preço da corrida, o tempo de espera, a rota mais eficiente, etc.

Nos bastidores deste aplicativo existem diversos recursos que utilizam IA, como por exemplo:
- Sistemas de NLP (Natural Language Processing) para entender o que o usuário está digitando no campo de busca.
- Software de visão computacional para identificar a localização do usuário.
- Algoritmos de processamento de sensores que ajudam a melhorar a precisão em áreas urbanas, podendo incluir até mesmo uma identificação automática de acidentes.
- Algoritmos de machine learning para prever a demanda de corridas em determinadas áreas e horários.

## Capítulo 1 - Fundamentos da IA

### Alan Turing e o teste de Turing:

Através do seu artigo "Computing Machinery and Intelligence", na qual concentrou-se no conceito de uma máquina inteligente, Turing buscou um meio de avaliar a inteligência de uma máquina. Para isso, ele propôs um teste, que ficou conhecido como "teste de Turing", no qual um juiz humano interage com dois participantes, um humano e uma máquina, através de um terminal de computador. O juiz não sabe quem é o humano e quem é a máquina, e deve tentar descobrir quem é quem através de perguntas. Se o juiz não conseguir distinguir entre o humano e a máquina, então a máquina é considerada inteligente.

Em 2014, um programa de computador chamado Eugene Goostman, que simulava um garoto de 13 anos, passou no teste de Turing, fazendo com que muitos acreditassem que a IA havia chegado ao seu ápice. Porém, o teste de Turing não é um teste definitivo, pois ele não é capaz de avaliar a inteligência de uma máquina, apenas a capacidade de simular um humano. Durante o teste, o programa foi capaz de enganar cerca de 33% dos juizes, um dos fatores que contribuiu para isso foi o fato de que o programa simulava um garoto de 13 anos, e não um adulto. O programa era baseado em simular um humano, e não em ser inteligente.

Já em 2018 o Google apresentou o Google Duplex, um sistema de IA capaz de realizar ligações telefônicas para agendar compromissos, e que foi capaz de enganar os humanos do outro lado da linha, que não perceberam que estavam falando com uma máquina. Mas ele ainda não foi capaz de passar no teste de Turing, pois ele não é capaz de responder perguntas abertas, apenas de realizar tarefas específicas.

Há muita controvérsia, entretanto sobre o teste de turing, sugerindo até mesmo uma manipulação dos resultados. Em 1980, o filósofo John Searle escrever um artigo famoso, "Minds, Brains and Programs", no qual descrever seu proprio argumento de pensamento, o "Argumento do quarto chinês". Nesse argumento, ele propós colocar uma pessoa X em uma sala que não sabe falar o idioma chinês, mas possuí um livro que contém manuais extremamente simples de como ele poderia traduzir qualquer pergunta feita em chinês para uma resposta em chinês. Então, uma pessoa Y faz uma pergunta em chinês para a pessoa X, que utiliza o livro para traduzir a pergunta e responder em chinês. A pessoa Y não sabe que a pessoa X não fala chinês, e então acredita que a pessoa X fala chinês. Nesse caso, a pessoa X é a máquina, o livro é o programa de computador, e a pessoa Y é o juiz. O argumento de Searle é que a pessoa X não fala chinês, apenas está seguindo instruções, e por isso não é inteligente.

Searle também propós duas formas de IA:
- IA forte: Quando uma máquina compreende o que está acontecendo, podendoe até mesmo existir emoções e criatividade. Também chamada de inteligência artificial geral.
- IA fraca: A máquina realiza tarefas específicas, mas não compreende o que está acontecendo. Também chamada de inteligência artificial estreita.

Outras alternativas de testes também foram propostas, como por exemplo o teste de Lovelace, que propós que uma máquina só pode ser considerada inteligente se ela for capaz de criar algo original, e não apenas simular um humano.
Também tem o teste do café, na qual um robô deve ser capaz de entrar na casa de um estranho, localizar a cozinha  e preparar uma xícara de café.

### O cérebro é uma máquina?

Em 1943, Warren McCulloch e Walter Pitts publicaram um artigo chamado "A Logical Calculus of the Ideas Immanent in Nervous Activity", no qual descreveram um modelo de neurônio artificial, que foi chamado de neurônio de McCulloch-Pitts. Esse modelo foi baseado em um neurônio biológico, que é composto por um corpo celular, dendritos e axônios. Os dendritos recebem sinais de outros neurônios, e o axônio envia sinais para outros neurônios. O corpo celular processa os sinais recebidos e envia sinais para o axônio. O neurônio de McCulloch-Pitts é composto por um conjunto de entradas, um conjunto de pesos e uma função de ativação. As entradas são os sinais recebidos, os pesos são os valores que são atribuídos a cada entrada, e a função de ativação é a função que determina se o neurônio será ativado ou não. O neurônio de McCulloch-Pitts é um modelo simplificado de um neurônio biológico, e é um dos modelos mais simples de neurônio artificial.

A tese era que as funções principais do cérebro poderiam ser explicadas por meio de lógica booleana, com operadores de E, OU e NÃO.

### Cibernética:

Em 1948, Norbert Wiener publicou "Cybernetic:Or Controland Communication in the Animal and the Machine" no qual ele descreveu a cibernética como o estudo do controle e comunicação em máquinas e animais. Ele também descreveu a cibernética como a ciência da comunicação e controle, e que a cibernética poderia ser aplicada em diversas áreas, como por exemplo na medicina, na economia, na psicologia, na sociologia.

Os temas abordados no artigo eram bastante diversos. O livro foi uma atencipação da teoria do caos, e também descreveu a teoria dos sistemas, que é a ideia de que um sistema é composto por diversos componentes que interagem entre si.

Era especulado também que um computador poderia jogar xadrez e vencer um humano, e que um computador poderia ser capaz de aprender com o ambiente.

Ele chegou a pensar que as máquinas poderiam tornar as pessoas desnecessárias, e que as máquinas poderiam se tornar mais inteligentes que os humanos.

Ele criou diversas teórias, mas a mais famosa era relacionada a cibernética, na qual ele demonstrava a importância dos loops de feedback, atráves da compreensão do controle e das comunicações.

### História da Origem:

O termo "inteligência artificial" foi criado em 1956, por John McCarthy, que organizou uma conferência na qual o termo foi utilizado pela primeira vez. A conferência foi chamada de "The Dartmouth Summer Research Project on Artificial Intelligence", e foi organizada por McCarthy, Marvin Minsky, Nathaniel Rochester e Claude Shannon. A conferência foi um marco para a IA, pois foi a primeira vez que o termo foi utilizado, e também foi a primeira vez que o termo "programação" foi utilizado.

O objetivo da conferência era criar uma máquina que fosse capaz de simular a inteligência humana, e que fosse capaz de aprender com o ambiente. A conferência também foi responsável por criar o primeiro programa de IA, o Logic Theorist, que foi criado por Allen Newell, Herbert Simon e Cliff Shaw. O Logic Theorist foi capaz de provar teoremas matemáticos, e foi o primeiro programa de IA a ser capaz de aprender com o ambiente.

Eles utilzaram um computador IBM 701, que usava linguagem de máquina. Então, criaram uma linguagem de alto nível, chamada de IPL (Information Processing Language), que foi a primeira linguagem de programação de alto nível. A linguagem de programação foi criada por Newell, Simon e Shaw, e foi baseada na linguagem de programação Fortran.

O IBM não tinha memória suficiente para rodar o Logic Theorist, o que levou a uma nova inovação: Processamento de listas encadeadas. O programa foi capaz de rodar em 1957, e foi capaz de provar 38 dos 52 teoremas que foram propostos.

Apesar disso, o Logic Theorist não despertou muito interesse, a conferência foi um fracasso, e a IA foi considerada um fracasso. O motivo disso foi que o Logic Theorist não foi capaz de resolver problemas do mundo real, e também não foi capaz de aprender com o ambiente.


### McCarthy:

Eventos importantes:

-   No fim da decada de 1950, ele desenvolveu a linguagem de programação Lisp, que foi a primeira linguagem de programação funcional, e que foi baseada no cálculo lambda. O pesquisador também criou conceitos como recursão, tipagem dinâmica e a coleta de lixo.

-  Em 1961 criou o conceito de Time Sharing, que é a ideia de que um computador pode ser utilizado por diversos usuários ao mesmo tempo. Graças a esse conceito, levou ao desenvolvimento da internet e da computação em nuvem

### Era de Ouro:

A era de ouro da IA foi marcada entre 1956 a 1974.

Grande parte da inovação em IA aconteceu no circulo acadêmico.

Houve diversos programas que visavam alcançar algum nível de IA forte, como por exemplo o General Problem Solver, que foi criado por Newell e Simon, e que foi capaz de resolver problemas de lógica e de matemática.

Além disso houve outros como por exemplo:

-   ELIZA (1965): Foi criado por Joseph Weizenbaum, e foi capaz de simular um psicoterapeuta. O programa foi capaz de simular uma conversa com um psicoterapeuta, e foi capaz de enganar algumas pessoas. O programa foi baseado em padrões de palavras, e não em inteligência. Foi o primeiro exemplo de um chatbot.

-   SAINT (Symbolic Automatic INTegrator, 1961): Criado por James Slagle, foi capaz de resolver problemas de cálculo diferencial. Foi o primeiro exemplo de um sistema especialista.

-   Computer Vision (1966): Marvin Minsky disse a um estudante Gerald Jay Sussman, que passasse o verão com uma camera ligada a um computador e fizesse o que o computador descrevesse. O jovem construiu um sistema que detectou padrões básicos. Foi o primeiro exemplo de visão computacional.

-   Mac Hack (1968): Foi criado por Tom Evans, e foi capaz de jogar xadrez. Foi o primeiro exemplo de um programa de IA que jogava xadrez.

-   Hearsay(Fim da decada de 1960): Foi criado por Lee Erman, Victor Lesser e Daniel Bobrow, e foi capaz de reconhecer a voz humana. Foi o primeiro exemplo de reconhecimento de voz.

Havia, entretanto duas teorias principais sobre a inteligência artificial:

-   Simbólica: Proposta por Minsky , que disse que precisavam existir sistemas simbolicos. Ou seja a IA deveria se basear em lógica tradicional de um computador ou na pré-programação, ou seja estruturas de if, then, else

-   Conexão: Frank Rosenblatt, que disse que a IA deveria se basear em redes neurais, que são sistemas que simulam o cérebro humano. Em vez de chamar as partes internas de neurônios, ele chamou de perceptrons. O perceptron é um modelo de neurônio artificial, que é composto por um conjunto de entradas, um conjunto de pesos e uma função de ativação. As entradas são os sinais recebidos, os pesos são os valores que são atribuídos a cada entrada, e a função de ativação é a função que determina se o neurônio será ativado ou não. O perceptron é um modelo simplificado de um neurônio biológico, e é um dos modelos mais simples de neurônio artificial.

Em 1957 Rosenblatt criou o primeiro programa de computador seguindo essa óptica, chamou de Mark I Perceptron. O Mark I usou dados com ponderações aleatorias para treinar o modelo.

Isso foi inovador para a IA. Entretanto havia problemas persistentes com o perceptron. Um deles era que a rede neural tinha apenas uma camada. O outro era que a pesquisa do cérebro ainda estava nos estagios iniciais e não oferecia muitas informações.

Minsky e Papert publicaram um livro chamado "Perceptrons: An Introduction to Computational Geometry", no qual eles descreveram as limitações do perceptron, e que a IA não poderia ser baseada em redes neurais.

Entretanto, na decada de 1980, a IA baseada em redes neurais voltou a ganhar força, principalmente com o desenvolvimento do deep learning.

### Inverno da IA:

Durante o inicio da decada de 1970, o entusiasmo com a IA começou a diminuir, perdurou pela decada de 1980, e foi chamado de "Inverno da IA".

O motivo disso foi que os programas de IA não estavam sendo capazes de resolver problemas do mundo real, e também não estavam sendo capazes de aprender com o ambiente.

Além disso, os computadores da época não tinham memória suficiente para rodar os programas de IA, e também não tinham poder de processamento suficiente.

A linguagem LISP não era ideal para sistemas computacionais, foco no mundo corporativo estava voltado para o FORTRAN.

Tem-se também a questão econômica, pois a IA era muito cara, e não havia um retorno financeiro, além disso houve inflação persistente na decada de 1970.

Um professor Sir James Lighthill, publicou em 1973 um relatório que era um repúdio aos objetivos da IA forte. Esse relatório levou a um debate público sobre a IA, e também levou a uma redução de verbas para a pesquisa em IA.

Entretanto, mesmo no inverno da IA, houve alguns avanços, como por exemplo:

-   Backpropagation (1974): Foi criado por Paul Werbos, e foi capaz de treinar redes neurais. Foi o primeiro exemplo de um algoritmo de treinamento de redes neurais.

-   MYCIN (1976): Foi criado por Edward Shortliffe, e foi capaz de diagnosticar doenças. Foi o primeiro exemplo de um sistema especialista.

-   Deep Blue (1997): Foi criado por Feng-hsiung Hsu, e foi capaz de vencer o campeão mundial de xadrez. Foi o primeiro exemplo de um programa de IA que venceu um humano em um jogo de tabuleiro.


### Ascensão e queda dos sistemas especialistas:

Se baseavam nos conceitos de lógica simbolica de Minsky.

Um dos motivos primordiais para o impulso desse sistema foi o boom dos minicomputadores e PCs.

Um dos problemas dos sistemas especialistas e que eram muitos especificios e era dificil aplica-los em outras categorias.

No final da decada de 1980 os sistemas especialistas começaram a perder força, pois os computadores começaram a ter mais poder de processamento e memória, e também começaram a ter mais espaço de armazenamento. Isso levou a um novo inverno da IA.

### Redes Neurais e Deep Learning:

Geoffrey Hinton acreditou que o caminho do Rosenblatt era o caminho certo.

Ele também percebeu que o maior obstaculo à IA era a falta de poder computacional. Entretanto ele viu que o tempo estava ao seu lado graças a lei de Moore.

Hinton, David Rumelhart e Ronald Williams publicaram um artigo chamado "Learning Representations by Back-Propagating Errors", no qual eles descreveram o algoritmo de backpropagation, que foi capaz de treinar redes neurais.

Esse trabalho foi um marco para a IA, pois foi capaz de resolver um dos maiores problemas da IA, que era o treinamento de redes neurais.

Ele estimulou e teve como base outros pesquisadores da época como:

-   Kunihiko Fukushima: Criou o Neocognitron, que foi capaz de reconhecer padrões visuais. O Neocognitron era composto por camadas de neurônios, e cada camada era responsável por reconhecer um padrão específico. A primeira camada era responsável por reconhecer padrões simples, como por exemplo linhas retas. A segunda camada era responsável por reconhecer padrões mais complexos, como por exemplo círculos. A terceira camada era responsável por reconhecer padrões ainda mais complexos, como por exemplo rostos. O Neocognitron foi o primeiro exemplo de uma rede neural convolucional.

-   Yann Lecun mesclou redes neurais convolucionais com backpropagation, e criou o LeNet, que foi capaz de reconhecer dígitos escritos a mão. O LeNet foi o primeiro exemplo de uma rede neural convolucional treinada com backpropagation.

-   Yann Lecun publicou um artigo chamado "Gradient-Based Learning Applied to Document Recognition", que utilizou algoritmos de descida de gradiente para treinar redes neurais convolucionais. Esse trabalho foi um marco para a IA, pois foi capaz de resolver um dos maiores problemas da IA, que era o treinamento de redes neurais convolucionais.

### No contexto moderno:

Um dos aspectos que vem impulsionando a IA no contexto atual, moderno é:

-   Crescimento explosivo de dados: A quantidade de dados gerados está crescendo exponencialmente, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de analisar grandes quantidades de dados, e extrair informações valiosas.

-   Poder computacional: Os computadores estão cada vez mais poderosos, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de realizar cálculos complexos em um curto espaço de tempo.

-   Infraestrutura de nuvem: A infraestrutura de nuvem está cada vez mais acessível, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar a infraestrutura de nuvem para realizar cálculos complexos.

-   Algoritmos de IA: Os algoritmos de IA estão cada vez mais sofisticados, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar algoritmos sofisticados para realizar cálculos complexos.

-   Aprendizado de máquina: O aprendizado de máquina está cada vez mais sofisticado, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar o aprendizado de máquina para realizar cálculos complexos.

### Estrutura da IA

Em uma visão de alto nível, podemos relacionar os termos IA com Machine Learning e Deep Learning.

Sendo a IA o termo mais amplo, que engloba o Machine Learning, que por sua vez engloba o Deep Learning.

## Capítulo 2 - Dados

#### O Combustível da IA

### Noções básicas de dados:

Dados são informações que podem ser armazenadas e processadas por um computador.

Um bit é a menor unidade de dados que um computador pode armazenar, e pode ter o valor de 0 ou 1.

Um byte é composto por 8 bits, e pode ter o valor de 0 a 255.

Tabela de medidas:

| Unidade | Valor | Caso de uso |
| --- | --- | --- |
| Bit | 0 ou 1 | Armazenar um único valor |
| Byte | 0 a 255 | Armazenar um caractere |
| Kilobyte | 1024 bytes | Armazenar um parágrafo de texto |
| Megabyte | 1024 kilobytes | Armazenar uma foto |
| Gigabyte | 1024 megabytes | Armazenar um filme |
| Terabyte | 1024 gigabytes | Armazenar uma biblioteca de livros |
| Petabyte | 1024 terabytes | Armazenar a coleção de livros da biblioteca do congresso dos EUA |
| Exabyte | 1024 petabytes | Armazenar todos os dados gerados pela humanidade em um ano |
| Zettabyte | 1024 exabytes | Armazenar todos os dados gerados pela humanidade em 100 anos |
| Yottabyte | 1024 zettabytes | Armazenar todos os dados gerados pela humanidade em 10.000 anos |


Os dados podem vir de diversas fontes, como por exemplo:

-   Dados estruturados: São dados que possuem uma estrutura definida, e que podem ser armazenados em um banco de dados relacional. Por exemplo, uma tabela de clientes de uma empresa, que possui os campos nome, idade, endereço, etc.

-   Dados não estruturados: São dados que não possuem uma estrutura definida, e que não podem ser armazenados em um banco de dados relacional. Por exemplo, um vídeo, uma foto, um áudio, etc.

-   Dados semiestruturados: São dados que possuem uma estrutura definida, mas que não podem ser armazenados em um banco de dados relacional. Por exemplo, um arquivo XML, um arquivo JSON, etc.

### Tipos de dados:

Há quatro maneiras de orgazinar os dados:

#### Dados estruturados:
São dados que possuem uma estrutura definida, e que podem ser armazenados em um banco de dados relacional. Por exemplo, uma tabela de clientes de uma empresa, que possui os campos nome, idade, endereço, etc. Exemplos de dados estruturados são: tabelas de banco de dados, arquivos CSV, arquivos XML, arquivos JSON, etc.

    Na maior parte das vezes é mais fácil trabalhar com dados estruturados, pois eles possuem uma estrutura definida, e podem ser armazenados em um banco de dados relacional.

São frequentemente provenientes de sistemas de CRM (Customer Relationship Management), ERP (Enterprise Resource Planning), e geralmente tem volume menor que os dados não estruturados.

#### Dados não estruturados:

São dados que não possuem uma estrutura definida, e que não podem ser armazenados em um banco de dados relacional. Por exemplo, um vídeo, uma foto, um áudio, etc. Exemplos de dados não estruturados são: vídeos, fotos, áudios, etc.

    Na maior parte das vezes é mais difícil trabalhar com dados não estruturados, pois eles não possuem uma estrutura definida, e não podem ser armazenados em um banco de dados relacional.

São frequentemente provenientes de redes sociais, e geralmente tem volume maior que os dados estruturados.

É mais trabalhoso devido a necessidade de formatar os dados para que possam ser utilizados.

#### Dados semiestruturados:

São dados que possuem uma estrutura definida, mas que não podem ser armazenados em um banco de dados relacional. Por exemplo, um arquivo XML, um arquivo JSON, etc. Exemplos de dados semiestruturados são: arquivos XML, arquivos JSON, etc.

    Na maior parte das vezes é mais difícil trabalhar com dados semiestruturados, pois eles não podem ser armazenados em um banco de dados relacional.

São frequentemente provenientes de sistemas de CRM (Customer Relationship Management), ERP (Enterprise Resource Planning), e geralmente tem volume menor que os dados não estruturados.

#### Dados temporais:

Podem ser estruturados, semiestruturados ou não estruturados.

    É mais confuso e dificil de entender.

Esse tipo de dado é para interações; como por exemplo: Rastrear os "passos de um cliente".

### Big Data:

Big Data é um termo que se refere a grandes volumes de dados, que podem ser estruturados, semiestruturados ou não estruturados. O termo Big Data é utilizado para descrever o grande volume de dados, a velocidade com que os dados são gerados, e a variedade de dados que são gerados.

Como forma de lidar com dados muito grande, criou-se o Big Data.

A Oracle explica essa importância da seguinte forma:

    Hoje, big data, tornou-se essencial. Pense em todas as coisas que você faz todos os dias e em quantos dados são gerados a partir dessas atividades. Cada vez que você faz uma compra, pesquisa na Internet, baixa músicas, faz login em um site ou aplicativo, ou até mesmo joga um jogo no seu celular, você gera dados. Se você multiplica isso por todos os usuários da Internet, então você começa a ter uma ideia de quão grande é o big data.

#### Os 3 Vs do Big Data:

-   Volume: Refere-se a escala dos dados, que muitas vezes não são estruturados. Por exemplo, um único tweet pode ser considerado um dado, e o Twitter gera cerca de 500 milhões de tweets por dia. O volume é um grande desafio para o Big Data, pois é necessário armazenar e processar grandes volumes de dados. A computação em nuvem e as bases de dados NoSQL são algumas das tecnologias que ajudam a lidar com o volume de dados.

-   Variedade:  Refere-se a diversidade dos dados, que podem ser estruturados, semiestruturados ou não estruturados. Por exemplo, um vídeo, uma foto, um áudio, etc. A variedade é um grande desafio para o Big Data, pois é necessário armazenar e processar diversos tipos de dados. A computação em nuvem e as bases de dados NoSQL são algumas das tecnologias que ajudam a lidar com a variedade de dados.

-   Velocidade: Refere-se a velocidade com que os dados são gerados. Serviços Web geram niveis extremos de dados em tempo real. Por exemplo, o Google processa cerca de 40.000 pesquisas por segundo, e o Facebook processa cerca de 500.000 comentários por minuto. A velocidade é um grande desafio para o Big Data, pois é necessário armazenar e processar dados em tempo real. Esse 'V' é o mais importante, pois é o que mais gera valor para as empresas.

Existem outros Vs que também são importantes:

-   Veracidade: Refere-se a qualidade dos dados, que muitas vezes são imprecisos. Por exemplo, um tweet pode conter informações falsas, e um vídeo pode conter informações falsas. A veracidade é um grande desafio para o Big Data, pois é necessário garantir a qualidade dos dados.

-   Valor: Refere-se ao valor dos dados, que muitas vezes não são utilizados. Por exemplo, um tweet pode conter informações valiosas, e um vídeo pode conter informações valiosas. O valor é um grande desafio para o Big Data, pois é necessário extrair informações valiosas dos dados.

-   Visualização: Refere-se ao uso de recursos visuais para representar os dados. Por exemplo, um gráfico, um mapa, etc. A visualização é um grande desafio para o Big Data, pois é necessário representar os dados de forma visual.

### Bancos de dados e outras ferramentas:

#### Bancos de dados relacionais:

Um banco de dados relacional é um banco de dados que armazena dados em tabelas, e que utiliza chaves primárias e chaves estrangeiras para relacionar as tabelas. Um banco de dados relacional é composto por um conjunto de tabelas, e cada tabela é composta por um conjunto de linhas e colunas. Cada linha representa um registro, e cada coluna representa um campo. Um banco de dados relacional é uma das formas mais comuns de armazenar dados estruturados.

Quando o big data começou a crescer, os bancos de dados relacionais começou a apresentar falhas graves, como:

-   Expansão de dados: Os bancos de dados relacionais não são capazes de armazenar grandes volumes de dados, e isso é um grande problema para o Big Data. Ficou mais dificil centralizar os dados.

-   Novos ambientes: Banco de dados relacionais não foram criados para computação em nuvem.

-   Custos: Os bancos de dados relacionais são muito caros, e isso é um grande problema para o Big Data.

Devido a isso, havia projetos de códi aberto que visavam contornar esse problema. Um deles foi o Hadoop, que foi criado por Doug Cutting e Mike Cafarella, e que foi capaz de armazenar e processar grandes volumes de dados. O Hadoop foi um marco para o Big Data, pois foi capaz de resolver um dos maiores problemas do Big Data, que era o armazenamento e processamento de grandes volumes de dados.

Houve também a inovação do modelo tradicional: o NoSQL.

### Data Lakes:

Um data lake é um repositório de dados que armazena dados em seu formato nativo, e que pode ser utilizado para armazenar dados estruturados, semiestruturados ou não estruturados. Um data lake é composto por um conjunto de dados, e cada dado é composto por um conjunto de linhas e colunas. Cada linha representa um registro, e cada coluna representa um campo. Um data lake é uma das formas mais comuns de armazenar dados estruturados, semiestruturados ou não estruturados.

O Data lake tratará as diversas fontes de dados, e irá armazenar em um único lugar.

### Processo de dados:

Muitos projetos de big data são abandonados antes de chegar na fase piloto. Alguams razões incluem:

-   Falta de habilidades: A falta de habilidades é um grande problema para o Big Data, pois é necessário ter conhecimento em diversas áreas, como por exemplo programação, estatística, etc.

-   Falta de dados: A falta de dados é um grande problema para o Big Data, pois é necessário ter dados para realizar análises.

-   Falta de recursos: A falta de recursos é um grande problema para o Big Data, pois é necessário ter recursos para realizar análises.

-   Falta de foco no negócio: A falta de foco no negócio é um grande problema para o Big Data, pois é necessário ter foco no negócio para realizar análises.

#### Processo CRISP-DM:

O CRISP-DM é um processo de mineração de dados que é composto por seis fases: entendimento do negócio, entendimento dos dados, preparação dos dados, modelagem, avaliação e implantação. O CRISP-DM é um dos processos mais utilizados para realizar análises de dados.

-   Entendimento do negócio: Nessa fase, é necessário entender o problema de negócio, e definir os objetivos do projeto. É necessário entender o problema de negócio, pois é necessário definir os objetivos do projeto.  Exemplo: "Aumentar as vendas em 10%".

-   Compreensão dos dados: Existem três fontes de dados principais para o projeto:
        
        - Dados internos: Dados que são gerados pela empresa, como por exemplo dados de vendas, dados de marketing, etc. As vantagens dos dados internos são que eles são fáceis de obter, e que eles são confiáveis. As desvantagens dos dados internos são que eles são limitados, e que eles são difíceis de analisar.

        - Dados de código aberto: Costumam estar disponíveis gratuitamente. Alguns exemplos são dados cientificos e governamentais. Esse tipo de dado ja vem formatado, entretanto, alguma das variaveis podem não ser claras

        - Dados de terceiros: Dados que são gerados por terceiros, como por exemplo dados de redes sociais, dados de sensores, etc. As vantagens dos dados de terceiros são que eles são abundantes, e que eles são variados. As desvantagens dos dados de terceiros são que eles são difíceis de obter, e que eles são imprecisos.

    Deve-se fazer as seguintes perguntas para avaliar os dados:
        
        - Os dados são suficientes?
        - Os dados são relevantes?
        - Os dados são de qualidade?
        - Os dados são acessíveis?

-   Preparação dos dados: Nessa fase, é necessário preparar os dados para a análise. É necessário preparar os dados para a análise, pois é necessário garantir a qualidade dos dados. As etapas da preparação dos dados são: limpeza dos dados, transformação dos dados, redução dos dados e integração dos dados. Medidas para limpar os dados:

        - Remover dados duplicados (deduplicação)
        - Remover dados inválidos (validação)
        - Remover dados inconsistentes (padronização)
        - Remover dados incompletos (imputação)
        - Remover dados irrelevantes (filtragem)
        - Remover dados desatualizados (atualização)
        - Remover dados discrepantes (detecção de outliers)
        - Dados devem ter definições claras (Consistência)
        - Codificação one-hot encoding

    Medidas para transformar os dados:

        - Converter dados de um formato para outro
        - Converter dados de uma unidade para outra
        - Converter dados de uma escala para outra
        - Converter dados de uma estrutura para outra

    Medidas para reduzir os dados:

        - Reduzir o número de registros
        - Reduzir o número de campos

    Medidas para integrar os dados:

        - Integrar dados de diferentes fontes
        - Integrar dados de diferentes formatos
        - Integrar dados de diferentes estruturas

### Ética e privacidade:

É preciso estar atento a privacidade dos dados, e a ética de como eles são utilizados.
### Qual o volume de dados necessário para IA?

Fenômeno de Hughes: "A quantidade de dados necessária para treinar um modelo de IA é inversamente proporcional ao poder de processamento do computador".

Maldiçãod a dimensionalidade: "Quanto maior a dimensionalidade dos dados, maior a quantidade de dados necessária para treinar um modelo de IA". Logo a quantidade de dados que se precisa generalizar é muito grande.

Em caso geral, quanto mais dados, melhor.


## Capítulo 3 - Aprendizado de Máquina (Machine Learning)

### O que é aprendizado de máquina?

Em 1959, Arthur Samuel definiu o aprendizado de máquina como o campo de estudo que dá aos computadores a habilidade de aprender sem serem explicitamente programados.

Ele programou um computador para jogar damas, e o computador foi capaz de aprender com o ambiente.

Através do jogo de damas ele demonstrou como o machine learning funciona: o computador aprende com o ambiente, e é capaz de melhorar seu desempenho ao longo do tempo.

Isso foi possível por conta de conceitos avançados de Estatistica e Probabilidade.

Em essência, trata-se de um processo de adoção de dados rotulados e busca de relações entre eles.

Exemplo: Ensinar um programa a reconhecer um gato. Para isso, é necessário fornecer ao programa uma grande quantidade de imagens de gatos, e dizer ao programa que essas imagens são de gatos. O programa irá analisar as imagens, e irá aprender a reconhecer um gato.

### Desvio padrão:

Mede a dispersão dos dados em relação a média.

Exemplo: Se a média de uma amostra é 10, e o desvio padrão é 2, um desvio padrão abaixo da media é 8, e um desvio padrão acima da media é 12.

### Distribuição normal:

Também chamada curva do sino.

Soma das probabilidades para uma variavel.

Exemplo: Pontuações de QI.

### Teorema de Bayes:

Abordagem comum na ánalise de doenças e testes de diagnóstico.

Exemplo: Se uma doença afeta 1 em cada 10.000 pessoas, e o teste de diagnóstico tem 99% de precisão, qual a probabilidade de uma pessoa ter a doença se o teste der positivo?

Formula: P(A|B) = P(B|A) * P(A) / P(B)

### Correlação:

Algoritmos de machine learning geralmente envolvem algum tipo de correlação.

Exemplo: Se a temperatura aumenta, a venda de sorvete aumenta.

Uma forma de descrever a correlação é através do coeficiente de correlação de Pearson, que é um número entre -1 e 1, e que mede a força da correlação.

-   Se o coeficiente de correlação de Pearson for 0, não há correlação.

-   Se o coeficiente de correlação de Pearson for >=0,7 E <=1 , há uma correlação positiva perfeita.(Se houver um aumento em uma variável, haverá um aumento na outra variável).

-   Se o coeficiente de correlação de Pearson for > -1 e <= 0,3 , há uma correlação negativa perfeita. (Se houver um aumento em uma variável, haverá uma diminuição na outra variável).

Entretanto é preciso ter cuidado com a correlação, pois ela não implica causalidade. Por exemplo: A taxa de divórcio em Maine está correlacionada com o consumo de margarina per capita.

### O que se pode fazer com o aprendizado de máquina?

-   Manutenção preditiva: É a manutenção que é realizada com base em dados, e que é utilizada para prever falhas em equipamentos. Por exemplo, uma empresa pode utilizar o aprendizado de máquina para prever falhas em equipamentos, e realizar a manutenção antes que ocorra uma falha.

-   Recrutamento: É o processo de seleção de candidatos, e que é utilizado para prever o desempenho de um candidato. Por exemplo, uma empresa pode utilizar o aprendizado de máquina para prever o desempenho de um candidato, e selecionar o candidato com o melhor desempenho.

-   Detecção de fraudes: É a detecção de fraudes que é realizada com base em dados, e que é utilizada para prever fraudes. Por exemplo, uma empresa pode utilizar o aprendizado de máquina para prever fraudes, e evitar que ocorra uma fraude.

-   Expereiência do cliente: É a experiência que é oferecida ao cliente, e que é utilizada para prever o comportamento do cliente. Por exemplo, uma empresa pode utilizar o aprendizado de máquina para prever o comportamento do cliente, e oferecer uma experiência personalizada.

### Processo de aprendizado de máquina:

É importante adotar uma abordagem sistemática, afim de evitar resultados equivocados.

É necessário um processo de dados, como: Eles estão dispersos? Ou Existem alguns padrões? Ou Eles são confiáveis? Caso sim, aprendizado de máquina é uma boa opção.

O objetivo do processo de machine learning é criar um modelo que se baseie em um ou mais algoritmos. Isso é alcançado por meio de um processo de treinamento, que envolve a utilização de dados de treinamento para ajustar os parâmetros do modelo.

#### Etapa # 1 - Ordenar os dados:

Se os dados forem classificados, pode haver uma distorção no modelo.

O algoritmo detectaria essa ordenação como um padrão.

Dessa forma é uma boa ideia embaralhar os dados.

#### Etapa # 2 - Escolha do Modelo:

Envolve a seleção de um algoritmo de aprendizado de máquina.

#### Etapa # 3 - Treinamento do Modelo:

Envolve a utilização de dados de treinamento para ajustar os pesos do modelo.

Exemplo: Se o modelo for uma regressão linear, o algoritmo de treinamento ajustará os coeficientes da regressão linear.

Aproximadamente 70% dos dados são usados para criar relações no algoritmo.

#### Etapa # 4 - Avaliação do Modelo:

É necessário reunir dados de teste para avaliar o modelo. Formados pelos 30% restantes dos dados.

Com os dados de teste é possível avaliar o desempenho do modelo.

Observação: os dados de teste não devem ser utilizados para treinar o modelo.
Ou seja, os dados de treinamento e teste não devem ser nusturados.

A precisão é uma medida de sucesso do algoritmo. Entretanto pode ser enganosa, é necessário outras abordagens, como por exemplo o teorema de Bayes.

#### Etapa # 5 - Sintonia fina do modelo:

É possível ajustar os valores dos parâmetros no algoritmo. Nessa etapa, a intenção é verificar se é possível melhorar o desempenho do modelo.

Ao realizar a sintonia fina do modelo, é necessário ter cuidado para não realizar overfitting, que é quando o modelo se ajusta muito bem aos dados de treinamento, mas não se ajusta bem aos dados de teste.

Nessa sintonia, também pode verificar a existência de hiperparametros, que são parametros que não são ajustados pelo algoritmo de treinamento, e que precisam ser ajustados manualmente.

### Aplicando Algoritmos:

-   O primeiro passo é a processagem de dados, que envolve a limpeza dos dados, a transformação dos dados, a redução dos dados e a integração dos dados. Com isso o computador poderá começar a aprender

#### Aprendizagem supervisionada:

Faz uso de dados rotulados.

Dados rotulados: São dados que possuem um rótulo, e que podem ser utilizados para treinar um modelo de aprendizado de máquina. Por exemplo, uma imagem de um gato, que possui o rótulo "gato", e que pode ser utilizada para treinar um modelo de aprendizado de máquina.

Na maioria dos casos torna a análise mais fácil.

Deve haver grandes quantiddes de dados, para gerar um modelo preciso.

Entretanto, um problema é que a maioria dos dados não são rotulados.

Em alguns casos é possível automatizar o processo de rotulação, como por exemplo o reconhecimento de voz.

Aplicações:

-  Detecção de conteudo improprio

-  Detecção de spam

-  Detecção de fraude

#### Aprendizagem não supervisionada:

Faz uso de dados não rotulados.

Abordagem mais comum e o agrupamento de dados(clustering).

O processo de clustering geralmente se inicia com suposições sobre os dados, seguidas de iterações dos calculos para melhores resultados.

A busca por itens de dados proximos pode ser feito através de diversos metodos quantitativos:

-   Distância euclidiana: É a distância entre dois pontos em um espaço euclidiano. Por exemplo, a distância entre dois pontos em um plano cartesiano.

-  Métrica de Similariedade do cosseno: usa-se um cosseno para medir o angulo.

-  Distância de Manhattan: É a distância entre dois pontos em um espaço euclidiano, e que é calculada pela soma dos valores absolutos das diferenças entre as coordenadas dos pontos. Por exemplo, a distância entre dois pontos em um plano cartesiano.

Aplicações:

-   Segmentação de clientes

-   Analise de sentimento


#### Aprendizagem por reforço:

Faz uso de dados de recompensa.

É um processo de tentativa e erro.

O algoritmo de aprendizado de máquina recebe um estado, e executa uma ação. O algoritmo de aprendizado de máquina recebe uma recompensa, e aprende se a ação foi boa ou ruim.

Aplicações:

-   Jogos

-   Robótica


#### Aprendizagem semi-supervisionada:

Mistura de aprendizagem supervisionada e não supervisionada.

Surge quando há uma pequena quantidade de dados não rotulados.

É possível usar deep learning para transformar os dados não supervisionados em dados supervisionados, processo chamado de pseudorrotulagem.

Aplicações:

-   Reconhecimento de fala

-   Ressonância magnética

![ALgoritmos](/Livros/Introdução%20à%20Inteligência%20Aritificial:%20Uma%20abordagem%20não%20técnica%20-%20Tom%20Taulli/images/algoritmoMachineLearning.jpeg)

Todos os créditos da imagem para o autor do livro, o único intuito é de estudo e uma melhor compreensão do conteúdo.

#### Classificador: Naive Bayes(Aprendizagem supervisionada/Classificação):

Baseado em uma "suposição ingênua" de que as variáveis são independentes.

Ou seja, uma variável não afeta a outra.

Três variações do classificador Naive Bayes:

-   Naive Bayes Gaussiano: É utilizado quando as variáveis são contínuas. Por exemplo, a altura de uma pessoa.

-   Naive Bayes Multinomial: É utilizado quando as variáveis são discretas. Por exemplo, o número de filhos de uma pessoa.

-   Naive Bayes Bernoulli: É utilizado quando as variáveis são binárias. Por exemplo, se uma pessoa possui um carro ou não.

Casos comuns de uso desse algoritmo é para analise de texto, como por exemplo: classificação de spam, classificação de documentos, etc.

A abordagem deste algoritmo é útil na classificação de dados com base em características-chave e padrões.

Exemplo:

Imagine a seguinte tabela de vendas:

| Desconto | Avaliação do produto | Compra
| --- | --- | --- |
| Sim | Alta | Sim |
| Sim | Baixa | Sim |
| Não | Baixa | Não |
| Não | Baixa | Não |
| Não | Baixa | Não |
| Não | Alta | Sim |
| Sim | Alta | Não |
| Sim | Baixa | Sim |
| Não | Alta | Sim |
| Sim | Alta | Sim |
| Não | Alta | Não |
|Não | Baixa | Sim |
| Sim | Alta | Sim |
| Sim | Baixa | Não |

Organizando os dados em uma tabela de frequência:

| Desconto | Sim | Não | Total |
| --- | --- | --- | --- |
| Sim | 5 | 3 | 8 |
| Não | 3 | 3 | 6 |
| Total | 8 | 6 | 14 |

| Avaliação do produto | Alta | Baixa | Total |
| --- | --- | --- | --- |
| Sim | 5 | 3 | 8 |
| Não | 1 | 5 | 6 |
| Total | 6 | 8 | 14 |


| Compra | Sim | Não | Total |
| --- | --- | --- | --- |
| Sim | 7 | 1 | 8 |
| Não | 1 | 5 | 6 |
| Total | 8 | 6 | 14 |

Agora é possível calcular a probabilidade de compra com desconto:

P(Compra=Sim|Desconto=Sim) = P(Desconto=Sim|Compra=Sim) * P(Compra=Sim) / P(Desconto=Sim)

P(Desconto=Sim|Compra=Sim) = 7/8 = 0,875

P(Compra=Sim) = 8/14 = 0,571

P(Desconto=Sim) = 8/14 = 0,571

P(Compra=Sim|Desconto=Sim) = 0,875 * 0,571 / 0,571 = 0,875

Dessa forma observamos que o Naive Bayes é um algoritmo simples, e que é capaz de realizar cálculos complexos, sua funcionalidade é baseada em probabilidade.

Em termos gerais o algoritmo:

-   Calcula a probabilidade de cada classe.

-   Calcula a probabilidade de cada classe dado os valores dos atributos.

-   Calcula a probabilidade de cada classe dado os valores dos atributos e a classe.

Pseudo-código:

```

Para cada classe:

    Calcule a probabilidade da classe.

    Para cada atributo:

        Calcule a probabilidade do atributo.

        Calcule a probabilidade do atributo dado a classe.

    Calcule a probabilidade da classe dado os atributos.

```

#### K-Nearest Neighbors(Aprendizagem supervisionada/Classificação):

O Metódo k-Nearest Neighbor (k-NN) é usado para classificar um conjunto de dados (k é um número inteiro positivo, tipicamente pequeno, que representa o número de vizinhos mais próximos a serem considerados).

A ideia central é que valores próximos podem ser classificados juntos.

Não há nenhum processo de treinamento com os dados

O algoritmo é baseado em distância, e é necessário definir uma métrica de distância, como por exemplo a distância euclidiana.

Pseudo-código:

```

Para cada ponto no conjunto de dados:

    Calcule a distância entre o ponto e os outros pontos no conjunto de dados.

    Selecione os k pontos mais próximos.

    Atribua o ponto à classe que é mais frequente entre os k pontos mais próximos.

```

Caso exista dados categóricos , é possível utilizar uma métrica de sobreposição.

Grandes quantidades de vizinhos é bom para regular o modelo, entretanto pode ser ruim para a performance, o ideal é regular os pesos dos vizinhos, quanto mais próximo o vizinho, maior o peso.

#### Regressão Linear(Aprendizagem supervisionada/Regressão):

Indica a relação entre certas variaveis

Pode ajudar a prever os resultados com base nas entradas.

Exemplo:

Imagine a seguinte tabela de estudo e notas:

| Horas de estudo | Nota |
| --- | --- |
| 1 | 0,75 |
| 1 | 0,69 |
| 1 | 0,71 |
| 3 | 0,82 |
| 3 | 0,83 |
| 4 | 0,86 |
| 5 | 0,85 |
| 5 | 0,89 |
| 5 | 0,84 |
| 6 | 0,91 |
| 6 | 0,92 |
| 7 | 0,95 |

Temos uma relação positiva (quanto mais horas de estudo, maior a nota).

A regressão linear é uma abordagem que é utilizada para prever um valor contínuo, e que é baseada em uma relação linear entre uma variável dependente e uma ou mais variáveis independentes.

Com o algorithm de regressão linear é possível prever a nota com base nas horas de estudo.

Podemos obter a seguinte equação dessa relação:

Nota = Numero de horas de estudo x 0,03731 + 0,6889

O valor 0,03731 é o coeficiente de regressão, e o valor 0,6889 é o intercepto.

Para se obter o coeficiente de regressão e o intercepto, é necessário minimizar o erro quadrático médio.

O erro quadrático médio é a soma dos quadrados dos erros dividida pelo número de amostras.

Com essas informações poderiamos determinar, que nota um aluno teria se estudasse 10 horas.

Nota = 10 x 0,03731 + 0,6889 = 1,07

Esse modelo é bem simplista, para ter algo mais próximo da realidade é necessário mais variaveis, como por exemplo a frequencia do aluno. Nesse caso temos uma regressão linear multivariada.

Pseudo-código:

```

Para cada atributo:

    Calcule o coeficiente de regressão.

    Calcule o intercepto.

    Calcule o erro quadrático médio.

```

#### Árvore de decisão(Aprendizagem supervisionada/Classificação):

Essa abordagem é melhor com dados não numericos.

A arvore se iniciaria com uma raiz, e a partir dela se ramificaria em outras raizes, e assim por diante. A partir da raiz existe uma arvore de caminhos de decisão.

O mais famoso caso é referente ao naufraugio do Titanic, onde a arvore de decisão foi utilizada para determinar quem sobreviveria.

Exemplo:

                     É do sexo masculino?
                        /     \
                       /       \
                      /         \
                    Sim         Não
                    /             \
                   /               \
                Idade > 9,5?      Sobreviveu
                     /     \
                    /       \
                   /         \
                   Morreu    Quantidade de filhos > 2,5?
                                /     \
                               /       \
                              /         \
                            Sim         Não
                            /             \
                           /               \
                          /                 \
                         Morreu            Sobreviveu


As arvores de decisão são simples de entender e funcionam bem com grandes conjungos de dados e fornecem transparencia.

Entretanto se uma decisão se mostrar equivocada, ocorrera grande propagação de erro.

A medida que a arvore cresce, o desempenho diminui.

Pseudo-código:

```

Para cada atributo:

    Calcule a entropia.

    Calcule a informação mútua.

    Calcule o ganho de informação.

```

#### Modelagem por agrupamento(APrendizagem supervisionada/Regressão):

Usa mais de um modelo para prever os resultados.

Gera resultados mais precisos.

O melhor exemplo foi o Netflix Prize, onde o objetivo era melhorar o algoritmo de recomendação de filmes.

O algoritmo de recomendação de filmes da Netflix é baseado em agrupamento.

#### Agrupamento K-Means(Aprendizagem não supervisionada/Agupamento):

Eficiente em grandes conjuntos.

Coloca dados semelhantes não rotulados em diferentes grupos.

O primeiro passo é selecionar k, numero de grupos (clusters)

Exemplo:

        x     x
        x   x           x
                                x
        x    x x             x
                    x

Para esse exemplo, temos que haveŕa dois grupos, o que significa dois centros.


    o
        x     x
        x   x           x           o
                                x
        x    x x             x
                    x

O algoritmo calcularia a distancia entre os pontos e os centros, e os agruparia.

    o           |
        x     x |
        x   x   |        x           o
                |                x
        x    x x|             x
                |    x


Quanto maior o valor de k (numero de grupos), maior a precisão, entretanto maior o custo computacional. Ocorre apenas melhorias incrementais.

O algoritmo não funciona bem com dados não esfericos.

Há situações onde existem grupos com poucos dados, e outros com muitos dados, isso pode ocasionar em o algoritmo não selecionar os grupos corretamente.

Pseudo-código:

```

Para cada ponto no conjunto de dados:

    Calcule a distância entre o ponto e os centros.

    Atribua o ponto ao centro mais próximo.

    Recalcule os centros.

```

### Capítulo 4 - Deep Learning

#### Diferenças entre Machine Learning e Deep Learning:

Suponha um problema de reconhecimento de imagem, onde o objetivo é reconhecer um gato em uma imagem.

Há milhares de imagens de animais, e o objetivo é criar um modelo que seja capaz de reconhecer gatos em imagens.

Machine Learning não pode analisar as imagens diretamente, os dados devem ser rotulados. Uma possível forma de se fazer reconhecimento dos gatos seria com o aprendizado supervisionado, onde o algoritmo de aprendizado de máquina recebe uma imagem de um gato, e aprende que essa imagem é de um gato.

Entretanto, mesmo que ele consiga reconhecer um bom numero de gatos, ele não será capaz de reconhecer todos os gatos, pois ele não consegue reconhecer padrões complexos.

Dessa forma uma ánalise de pixel a pixel seria mais eficiente. Para fazer uma analise de pixel a pixel com o machine learning é necessário fazer a extração de recursos. Ou seja é necessário identificar os tipos de recursos que podem ser utilizados para reconhecer um gato em uma imagem, como por exemplo o formato das orelhas, o formato dos olhos, etc.

Entretanto essa abordagem por mais que seja boa, não é perfeita, pois caso
os dados sejam imprecisos, ou até mesmo ocorra exceções, o algoritmo não será capaz de reconhecer o gato.

Além disso a extração de recursos ignora uma boa quantidade de dados.

Deep Learning é uma abordagem que é utilizada para resolver esse problema, pois ela é capaz de analisar os dados de forma mais profunda, e é capaz de reconhecer padrões complexos.

O deep learning analisaria pixel por pixel, e encontraria relações usando uma 
rede neural.

#### O que é deep learning?

É uma subárea do machine learning, e que é utilizada para resolver problemas complexos.

Permite o processamento de enormes quantidade de dados visando encontrar padrões complexos.

Deep se refere a profundidade, e a profundidade se refere a quantidade de camadas ocultas (hidden layer) que uma rede neural possui.

Entretanto essa tecnólogia está nos estados iniciais, e ainda não é capaz de resolver todos os problemas.

Para que ele cause um impacto significativo, é necessário um crescimento exponencial de dados, e um aumento na capacidade de processamento.

#### Redes neurais artificiais:

São modelos computacionais que são inspirados no cérebro humano, e que são utilizados para resolver problemas complexos.

Em termos simples, é uma função com unidades(perceptrons, neuronios) de entrada, que são conectadas a unidades de saida, e que são utilizadas para resolver problemas complexos. Essas unidades estão atreladas a pesos, que são ajustados durante o processo de treinamento. Além disso é possível estarem ligadas a uma camada oculta, que é uma camada intermediária entre a camada de entrada e a camada de saida.

Também existe o viés(bias) que é um valor que é adicionado a entrada de uma função de ativação, e que é utilizado para ajustar a saida da função de ativação.

Esse tipo é chamado de rede neural feedforward, pois os dados fluem em uma direção, da camada de entrada para a camada de saida.

Uma saída pode ser a entrada de um novo perceptron, e assim por diante.

Ex de rede neural:

    1 b -
           \

    x1  w1 ----- Camada oculta ------ Saída 

              /
    x2 w2 -

Onde:

-   x1 e x2 são as entradas.

-   w1 e w2 são os pesos.

-   b é o viés.


Observe que ele se move em uma direção, da camada de entrada para a camada de saida, o ciclo não se repete.


#### Backpropagation:

É um algoritmo que é utilizado para treinar redes neurais artificiais.

Sua utilização conseguiu tratar um dos grandes problemas das redes neurais: O ajuste dos pesos. Seu funcionamento é baseado em ajustar a rede neural quando erros são encontrados. Ele calcula o erro, e ajusta os pesos para reduzir o erro. São varias iteracoes até que o erro seja minimizado.

Por exemplo, suponhamos que uma das entradas tenha saída igual a 0,6, o erro então é de 0,4, e o objetivo é reduzir esse erro. Para isso é necessário ajustar os pesos, e o viés. O objetivo é chegar o mais próximo possível de 1.

Se pensarmos em um gráfico de Erro no eixo Y e Peso no eixo X, nota-se que no começo tem-se um grande numero de erros e com as iterações do algoritmo o erro
diminui até encontrar um ponto de mínimo. Passado o ponto de mínimo, o erro aumenta, e o algoritmo para.

#### Redes neurais recorrentes:

A função processa a entrada atual e a entrada anterior.

É utilizada para analisar dados sequenciais, como por exemplo dados de texto, dados de audio, etc.

O melhor exemplo é a sugestão de texto do celular, onde o celular sugere palavras com base nas palavras anteriores.

Entretanto, existe o problema da dissipação do gradiente, que é quando o gradiente se torna muito pequeno, e o algoritmo não consegue aprender.

Os modelos tambem levam muito tempo para serem treinados.

O Google criou o Transformer, que é um modelo que é utilizado para tradução de texto, e que é baseado em redes neurais recorrentes.

Esse modelo processa em paralelo.

#### Redes neurais convolucionais:

São redes neurais que são utilizadas para analisar dados de imagem.

Existem casos onde as redes neurais não serem conectadas é interessante, por exemplo no caso de imagens, onde cada pixel é uma entrada, e cada pixel não está relacionado com o outro. Caso as redes neurais fossem conectadas, o algoritmo teria que analisar cada pixel, e isso seria muito custoso computacionalmente causando overfitting ( quando o modelo se ajusta muito bem aos dados de treinamento, mas não se ajusta bem aos dados de teste).

As redes neurais convolucionais são baseadas em filtros, que são utilizados para analisar os dados.

O filtro é uma matriz que é utilizada para analisar os dados, e que é deslizada sobre os dados.

O filtro é utilizado para extrair recursos, e é utilizado para reduzir a dimensionalidade dos dados.

Ela deve ser capaz de reconhecer com precisão os recursos, e deve ser capaz de generalizar os recursos.

#### Redes Adversistárias Generativas:

São redes neurais que são utilizadas para gerar dados.

Se baseia na Teoria dos Jogos, onde há dois jogadores: O gerador e o discriminador.

O gerador é responsável por gerar dados, e o discriminador é responsável por determinar se os dados são reais ou falsos.

O gerador da rede adversária generativa é treinado para enganar o discriminador, e o discriminador é treinado para não ser enganado pelo gerador.

Ocorre então um aprendizado em conjunto, onde o gerador aprende a gerar dados mais realistas, e o discriminador aprende a determinar se os dados são reais ou falsos.

Essa tecnólogia foi revolucionária, pois é possível gerar dados realistas, como por exemplo imagens de pessoas que não existem.

É possível com isso gerar um vídeo de um famoso dizendo coisas que ele nunca disse.

Observe, entretanto, que essa tecnólogia pode ser utilizada para gerar dados falsos, como por exemplo um vídeo de um político dizendo coisas que ele nunca disse.

## Terminologia:

- Geocerca (geofencing): É uma área virtual que pode ser definida por um conjunto de coordenadas geográficas. É utilizada para delimitar uma área de interesse, e quando um dispositivo entra ou sai dessa área, um alerta é disparado.

- Outlier: É um valor que foge do padrão, e que pode ser considerado um erro ou uma anomalia. Por exemplo, um valor que é muito maior ou muito menor que os outros valores.

- one-hot encoding: É uma técnica que é utilizada para transformar dados categóricos em dados numéricos. Por exemplo, uma variável que possui os valores "vermelho", "verde" e "azul" pode ser transformada em três variáveis que possuem os valores 0 ou 1.

- Dados categóricos: dados que nçao tem um signifiado numérico, como por exemplo cores, nomes, etc. Mas que podem ser transformados em dados numéricos.

- Tipo de dado: É um atributo que define o tipo de dado que pode ser armazenado em uma variável. Por exemplo, uma variável do tipo inteiro só pode armazenar números inteiros.

- Analise descritiva: É uma análise que é utilizada para descrever os dados, e que é utilizada para responder perguntas como "O que aconteceu?" e "O que está acontecendo?".

- ETL (Extract, Transform, Load): É um processo que é utilizado para extrair dados de diversas fontes, transformar os dados em um formato comum, e carregar os dados em um data warehouse.

- Warehouse: É um repositório de dados que armazena dados em seu formato nativo, e que pode ser utilizado para armazenar dados estruturados, semiestruturados ou não estruturados.

- Recurso: coluna de um conjunto de dados.

- Instancia: linha de um conjunto de dados.

- Meta dados: dados sobre os dados.

- Olap: Online Analytical Processing, é um processo que é utilizado para analisar dados. Permite analisar dados de diversas perspectivas, e permite analisar grandes volumes de dados.

- Analise preditiva: uso de dados para previsões. Dependem de algumas abordagens da IA como machine learning e deep learning.

- Crowdsourcing: É um processo que é utilizado para obter informações de um grande número de pessoas. Por exemplo, uma empresa pode utilizar o crowdsourcing para obter informações sobre um produto.