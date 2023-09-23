# Inteligência Artificial

## Esse repositorio conterá todos os estudos e projetos que desenvolverei para aprender sobre Inteligência Artificial.

O primeiro direcionamento de estudo será utilizando o livro: "Introdução à Inteligência Artificial: Uma abordagem não técnica" do autor Tom Taulli.

## Introdução à Inteligência Artificial: Uma abordagem não técnica - Tom Taulli

Data de publicação do livro -> 17 de Dezembro de 2019

### Sistema Inteligente x Sistema Esperto:

- Um sistema inteligente dispõe de informações e funcionalidades que empoderam o usuário a tomar decisões mais inteligentes.

- Um sistema inteligente é capaz de aprender com o usuário e com o ambiente.

- Um sistema esperto esconde o processo de tomada de decisão do usuário.

- Um sistema esperto é capaz de tomar decisões sem a necessidade de interação com o usuário.

- O usuario então é passivo no sistema esperto (ele simplesmente consome as informações dadas pela máquina), enquanto que no sistema inteligente o usuário é ativo.

### Introdução:

Inteligência artificial está presente em diversos aspectos do nosso dia a dia, por exemplo em sistemas de recomendação de filmes, músicas, livros, etc.
Um dos aplicativos que faz um uso muito bom de IA é o Uber, que utiliza IA para calcular o preço da corrida, o tempo de espera, a rota mais eficiente, etc.

Nos bastidores deste aplicativo existem diversos recursos que utilizam IA, como por exemplo:
- Sistemas de NLP (Natural Language Processing) para entender o que o usuário está digitando no campo de busca.
- Software de visão computacional para identificar a localização do usuário.
- Algoritmos de processamento de sensores que ajudam a melhorar a precisão em áreas urbanas, podendo incluir até mesmo uma identificação automática de acidentes.
- Algoritmos de machine learning para prever a demanda de corridas em determinadas áreas e horários.

### Capítulo 1 - Fundamentos da IA

#### Alan Turing e o teste de Turing:

Através do seu artigo "Computing Machinery and Intelligence", na qual concentrou-se no conceito de uma máquina inteligente, Turing buscou um meio de avaliar a inteligência de uma máquina. Para isso, ele propôs um teste, que ficou conhecido como "teste de Turing", no qual um juiz humano interage com dois participantes, um humano e uma máquina, através de um terminal de computador. O juiz não sabe quem é o humano e quem é a máquina, e deve tentar descobrir quem é quem através de perguntas. Se o juiz não conseguir distinguir entre o humano e a máquina, então a máquina é considerada inteligente.

Em 2014, um programa de computador chamado Eugene Goostman, que simulava um garoto de 13 anos, passou no teste de Turing, fazendo com que muitos acreditassem que a IA havia chegado ao seu ápice. Porém, o teste de Turing não é um teste definitivo, pois ele não é capaz de avaliar a inteligência de uma máquina, apenas a capacidade de simular um humano. Durante o teste, o programa foi capaz de enganar cerca de 33% dos juizes, um dos fatores que contribuiu para isso foi o fato de que o programa simulava um garoto de 13 anos, e não um adulto. O programa era baseado em simular um humano, e não em ser inteligente.

Já em 2018 o Google apresentou o Google Duplex, um sistema de IA capaz de realizar ligações telefônicas para agendar compromissos, e que foi capaz de enganar os humanos do outro lado da linha, que não perceberam que estavam falando com uma máquina. Mas ele ainda não foi capaz de passar no teste de Turing, pois ele não é capaz de responder perguntas abertas, apenas de realizar tarefas específicas.

Há muita controvérsia, entretanto sobre o teste de turing, sugerindo até mesmo uma manipulação dos resultados. Em 1980, o filósofo John Searle escrever um artigo famoso, "Minds, Brains and Programs", no qual descrever seu proprio argumento de pensamento, o "Argumento do quarto chinês". Nesse argumento, ele propós colocar uma pessoa X em uma sala que não sabe falar o idioma chinês, mas possuí um livro que contém manuais extremamente simples de como ele poderia traduzir qualquer pergunta feita em chinês para uma resposta em chinês. Então, uma pessoa Y faz uma pergunta em chinês para a pessoa X, que utiliza o livro para traduzir a pergunta e responder em chinês. A pessoa Y não sabe que a pessoa X não fala chinês, e então acredita que a pessoa X fala chinês. Nesse caso, a pessoa X é a máquina, o livro é o programa de computador, e a pessoa Y é o juiz. O argumento de Searle é que a pessoa X não fala chinês, apenas está seguindo instruções, e por isso não é inteligente.

Searle também propós duas formas de IA:
- IA forte: Quando uma máquina compreende o que está acontecendo, podendoe até mesmo existir emoções e criatividade. Também chamada de inteligência artificial geral.
- IA fraca: A máquina realiza tarefas específicas, mas não compreende o que está acontecendo. Também chamada de inteligência artificial estreita.

Outras alternativas de testes também foram propostas, como por exemplo o teste de Lovelace, que propós que uma máquina só pode ser considerada inteligente se ela for capaz de criar algo original, e não apenas simular um humano.
Também tem o teste do café, na qual um robô deve ser capaz de entrar na casa de um estranho, localizar a cozinha  e preparar uma xícara de café.

#### O cérebro é uma máquina?

Em 1943, Warren McCulloch e Walter Pitts publicaram um artigo chamado "A Logical Calculus of the Ideas Immanent in Nervous Activity", no qual descreveram um modelo de neurônio artificial, que foi chamado de neurônio de McCulloch-Pitts. Esse modelo foi baseado em um neurônio biológico, que é composto por um corpo celular, dendritos e axônios. Os dendritos recebem sinais de outros neurônios, e o axônio envia sinais para outros neurônios. O corpo celular processa os sinais recebidos e envia sinais para o axônio. O neurônio de McCulloch-Pitts é composto por um conjunto de entradas, um conjunto de pesos e uma função de ativação. As entradas são os sinais recebidos, os pesos são os valores que são atribuídos a cada entrada, e a função de ativação é a função que determina se o neurônio será ativado ou não. O neurônio de McCulloch-Pitts é um modelo simplificado de um neurônio biológico, e é um dos modelos mais simples de neurônio artificial.

A tese era que as funções principais do cérebro poderiam ser explicadas por meio de lógica booleana, com operadores de E, OU e NÃO.

#### Cibernética:

Em 1948, Norbert Wiener publicou "Cybernetic:Or Controland Communication in the Animal and the Machine" no qual ele descreveu a cibernética como o estudo do controle e comunicação em máquinas e animais. Ele também descreveu a cibernética como a ciência da comunicação e controle, e que a cibernética poderia ser aplicada em diversas áreas, como por exemplo na medicina, na economia, na psicologia, na sociologia.

Os temas abordados no artigo eram bastante diversos. O livro foi uma atencipação da teoria do caos, e também descreveu a teoria dos sistemas, que é a ideia de que um sistema é composto por diversos componentes que interagem entre si.

Era especulado também que um computador poderia jogar xadrez e vencer um humano, e que um computador poderia ser capaz de aprender com o ambiente.

Ele chegou a pensar que as máquinas poderiam tornar as pessoas desnecessárias, e que as máquinas poderiam se tornar mais inteligentes que os humanos.

Ele criou diversas teórias, mas a mais famosa era relacionada a cibernética, na qual ele demonstrava a importância dos loops de feedback, atráves da compreensão do controle e das comunicações.

#### História da Origem:

O termo "inteligência artificial" foi criado em 1956, por John McCarthy, que organizou uma conferência na qual o termo foi utilizado pela primeira vez. A conferência foi chamada de "The Dartmouth Summer Research Project on Artificial Intelligence", e foi organizada por McCarthy, Marvin Minsky, Nathaniel Rochester e Claude Shannon. A conferência foi um marco para a IA, pois foi a primeira vez que o termo foi utilizado, e também foi a primeira vez que o termo "programação" foi utilizado.

O objetivo da conferência era criar uma máquina que fosse capaz de simular a inteligência humana, e que fosse capaz de aprender com o ambiente. A conferência também foi responsável por criar o primeiro programa de IA, o Logic Theorist, que foi criado por Allen Newell, Herbert Simon e Cliff Shaw. O Logic Theorist foi capaz de provar teoremas matemáticos, e foi o primeiro programa de IA a ser capaz de aprender com o ambiente.

Eles utilzaram um computador IBM 701, que usava linguagem de máquina. Então, criaram uma linguagem de alto nível, chamada de IPL (Information Processing Language), que foi a primeira linguagem de programação de alto nível. A linguagem de programação foi criada por Newell, Simon e Shaw, e foi baseada na linguagem de programação Fortran.

O IBM não tinha memória suficiente para rodar o Logic Theorist, o que levou a uma nova inovação: Processamento de listas encadeadas. O programa foi capaz de rodar em 1957, e foi capaz de provar 38 dos 52 teoremas que foram propostos.

Apesar disso, o Logic Theorist não despertou muito interesse, a conferência foi um fracasso, e a IA foi considerada um fracasso. O motivo disso foi que o Logic Theorist não foi capaz de resolver problemas do mundo real, e também não foi capaz de aprender com o ambiente.


#### McCarthy:

Eventos importantes:

-   No fim da decada de 1950, ele desenvolveu a linguagem de programação Lisp, que foi a primeira linguagem de programação funcional, e que foi baseada no cálculo lambda. O pesquisador também criou conceitos como recursão, tipagem dinâmica e a coleta de lixo.

-  Em 1961 criou o conceito de Time Sharing, que é a ideia de que um computador pode ser utilizado por diversos usuários ao mesmo tempo. Graças a esse conceito, levou ao desenvolvimento da internet e da computação em nuvem

#### Era de Ouro:

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

#### Inverno da IA:

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


#### Ascensão e queda dos sistemas especialistas:

Se baseavam nos conceitos de lógica simbolica de Minsky.

Um dos motivos primordiais para o impulso desse sistema foi o boom dos minicomputadores e PCs.

Um dos problemas dos sistemas especialistas e que eram muitos especificios e era dificil aplica-los em outras categorias.

No final da decada de 1980 os sistemas especialistas começaram a perder força, pois os computadores começaram a ter mais poder de processamento e memória, e também começaram a ter mais espaço de armazenamento. Isso levou a um novo inverno da IA.

#### Redes Neurais e Deep Learning:

Geoffrey Hinton acreditou que o caminho do Rosenblatt era o caminho certo.

Ele também percebeu que o maior obstaculo à IA era a falta de poder computacional. Entretanto ele viu que o tempo estava ao seu lado graças a lei de Moore.

Hinton, David Rumelhart e Ronald Williams publicaram um artigo chamado "Learning Representations by Back-Propagating Errors", no qual eles descreveram o algoritmo de backpropagation, que foi capaz de treinar redes neurais.

Esse trabalho foi um marco para a IA, pois foi capaz de resolver um dos maiores problemas da IA, que era o treinamento de redes neurais.

Ele estimulou e teve como base outros pesquisadores da época como:

-   Kunihiko Fukushima: Criou o Neocognitron, que foi capaz de reconhecer padrões visuais. O Neocognitron era composto por camadas de neurônios, e cada camada era responsável por reconhecer um padrão específico. A primeira camada era responsável por reconhecer padrões simples, como por exemplo linhas retas. A segunda camada era responsável por reconhecer padrões mais complexos, como por exemplo círculos. A terceira camada era responsável por reconhecer padrões ainda mais complexos, como por exemplo rostos. O Neocognitron foi o primeiro exemplo de uma rede neural convolucional.

-   Yann Lecun mesclou redes neurais convolucionais com backpropagation, e criou o LeNet, que foi capaz de reconhecer dígitos escritos a mão. O LeNet foi o primeiro exemplo de uma rede neural convolucional treinada com backpropagation.

-   Yann Lecun publicou um artigo chamado "Gradient-Based Learning Applied to Document Recognition", que utilizou algoritmos de descida de gradiente para treinar redes neurais convolucionais. Esse trabalho foi um marco para a IA, pois foi capaz de resolver um dos maiores problemas da IA, que era o treinamento de redes neurais convolucionais.

#### No contexto moderno:

Um dos aspectos que vem impulsionando a IA no contexto atual, moderno é:

-   Crescimento explosivo de dados: A quantidade de dados gerados está crescendo exponencialmente, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de analisar grandes quantidades de dados, e extrair informações valiosas.

-   Poder computacional: Os computadores estão cada vez mais poderosos, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de realizar cálculos complexos em um curto espaço de tempo.

-   Infraestrutura de nuvem: A infraestrutura de nuvem está cada vez mais acessível, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar a infraestrutura de nuvem para realizar cálculos complexos.

-   Algoritmos de IA: Os algoritmos de IA estão cada vez mais sofisticados, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar algoritmos sofisticados para realizar cálculos complexos.

-   Aprendizado de máquina: O aprendizado de máquina está cada vez mais sofisticado, e isso é um dos fatores que está impulsionando a IA. A IA é capaz de utilizar o aprendizado de máquina para realizar cálculos complexos.

#### Estrutura da IA

Em uma visão de alto nível, podemos relacionar os termos IA com Machine Learning e Deep Learning.

Sendo a IA o termo mais amplo, que engloba o Machine Learning, que por sua vez engloba o Deep Learning.



## Terminologia:

- Geocerca (geofencing): É uma área virtual que pode ser definida por um conjunto de coordenadas geográficas. É utilizada para delimitar uma área de interesse, e quando um dispositivo entra ou sai dessa área, um alerta é disparado.