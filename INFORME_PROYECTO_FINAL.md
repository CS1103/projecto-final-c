# Informe de Proyecto Final

## Implementación de Redes Neuronales Artificiales en C++ para el Juego Pong

---

### Curso: Programación III - Teoría 1

### Integrantes:
- **Varela Villarreal, Yeimi Adelmar**
- **Guerrero Gutierrez, Nayeli Belén - (202410790)**
- **Medina Patrick** (completar apellidos y código)
- **Araoz, Fali** (completar apellidos y código)
- **Ignacio** (completar apellidos y código)
- **Guillaume** (completar apellidos y código)

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Objetivos](#objetivos)
3. [Investigación Teórica](#investigación-teórica)
   - 3.1 Historia y evolución de las NNs.
   - 3.2 Principales arquitecturas: MLP, CNN, RNN.
   - 3.3 Algoritmos de entrenamiento: backpropagation, optimizadores.
4. [Diseño e Implementación](#diseño-e-implementación)
5. [Arquitectura del Sistema](#arquitectura-del-sistema)
6. [Metodología de Desarrollo](#metodología-de-desarrollo)
7. [Resultados y Análisis](#resultados-y-análisis)
8. [Conclusiones](#conclusiones)
9. [Bibliografía](#bibliografía)

---

## Introducción

Este proyecto implementa un **framework completo de redes neuronales artificiales** usando el lenguaje de programación C++, aplicado al problema del aprendizaje automático en el juego Pong. La implementación no solo demuestra los fundamentos teóricos y prácticos de las redes neuronales, sino que también desarrolla cada componente sin dependencias (bibliotecas) externas especializadas en aprendizaje automático como TensorFlow o PyTorch. Ello para lograr una comprensión profunda de los algoritmos subyacentes.

Las redes neuronales artificiales son modelos computacionales inspirados en la estructura y funcionamiento del cerebro humano, diseñados para aprender patrones complejos a partir de datos (Haykin, 2009, p.1). En este proyecto, se implementa una **red neuronal feedforward multicapa** que procesa información de manera secuencial a través de capas interconectadas de neuronas artificiales.

Además, se incluye un proceso de entrenamiento que utiliza **retropropagación** para ajustar los pesos de las conexiones entre neuronas, optimizando así el rendimiento de la red en la tarea de jugar Pong (Rumelhart et al., 1986). Este enfoque no solo proporciona una base sólida en el aprendizaje automático, sino que también permite experimentar con diferentes arquitecturas de red, tasas de aprendizaje y funciones de activación, promoviendo una exploración activa de las técnicas de inteligencia artificial.

Al finalizar, se espera que los estudiantes no solo comprendan cómo funcionan las redes neuronales, sino que también sean capaces de aplicar estos conceptos a problemas más complejos en el ámbito del aprendizaje automático.

---

## Objetivos

### Objetivo General
Implementar un framework completo de redes neuronales artificiales en C++ sin dependencias externas especializadas, aplicado al aprendizaje automático en el juego Pong, para demostrar los fundamentos teóricos y prácticos de las redes neuronales.

### Objetivos Específicos

1. **Desarrollar un sistema de álgebra lineal propio** que incluya:
   - Implementación de tensores multidimensionales genéricos
   - Operaciones matriciales optimizadas (multiplicación, transposición)
   - Soporte para diferentes tipos de datos numéricos (float, double)

2. **Implementar arquitecturas de redes neuronales fundamentales**:
   - Capas densas (fully connected) con propagación hacia adelante y hacia atrás
   - Funciones de activación (ReLU, Sigmoid)
   - Funciones de pérdida (MSE, Binary Cross-Entropy)
   - Optimizadores (SGD, Adam)

3. **Crear un entorno de simulación completo**:
   - Implementar la física del juego Pong
   - Desarrollar un sistema de estados y recompensas
   - Crear una interfaz de visualización ASCII en tiempo real

4. **Desarrollar algoritmos de aprendizaje**:
   - Implementar el algoritmo de retropropagación desde cero
   - Aplicar técnicas de aprendizaje por refuerzo
   - Integrar estrategias epsilon-greedy para exploración/explotación

5. **Aplicar principios de ingeniería de software**:
   - Arquitectura orientada a objetos POO
   - Principios SOLID
   - Patrones de diseño (Strategy, Observer)

---

## Investigación Teórica

### 3.1 Historia y evolución de las NNs

#### Introducción Teórica
Las redes neuronales artificiales (RNA) constituyen la base del aprendizaje profundo y representan uno de los avances más significativos en inteligencia artificial. Inspiradas en el funcionamiento del cerebro humano, estas estructuras computacionales permiten resolver problemas complejos que los algoritmos tradicionales no pueden abordar eficientemente.

#### Orígenes y Desarrollo Histórico

 Dado el nacimiento de la informática moderna con la invención de la máquina de Turing en los 40s, la curiosidad por determinar lo que era computable o no eran la base de motivación de muchos autores por querer conocer la supremacía de la tecnología sobre los humanos. De ahí el test de Turing. Ambivalentemente, dos grandes como el neurólogo Warren McCulloch y el matemático Walter Pitts, motivados por la fuente de inteligencia humana, establecieron la primera arquitectura matemática de una neurona artificial. El concepto surgió dado este modelo.

Donald Hebb (1949) conocido también como el padre de la neuropsicología, dio a conocer el concepto abstracto de la "mente" con funciones cerebrales fisiológicas y biológicas específicas. Denominó el aprendizaje Hebbiano, definiendo que las neuronas forman redes y almacenan información en forma de recuerdos. "Las neuronas que se disparan juntas, se conectan" (Milner, 2003 p.1).

Frank Rosenblatt (1958) desarrolló el perceptrón, la primera red neuronal inspirado en el ojo de una mosca. Su importancia radica en ser un modelo fundamental para la clasificación de datos. Fue limitado, porque solo era efectivo para funciones lineales; esto fue demostrado por Marvin Minsky y Seymour Papert (1969) generando en la comunidad de conocedores una desilusión que terminó frenando la investigación sobre redes neuronales. Este tiempo se conoció como el "Invierno de la IA" que duró varias décadas entre finales de los 70 y finales de los 80 (Liu, 2024).

En 1986, David Rumelhart, Geoffrey Hinton y Ronald Williams publican un artículo que presenta el algoritmo de back-propagation. Con ello las investigaciones resurgieron. Este procedimiento permite el entrenamiento eficiente de redes neuronales multicapa al ajustar los pesos de las conexiones para minimizar la diferencia entre la salida real y la salida deseada (Rumelhart et al., 1986).

Avances como las redes convolucionales denominadas como CNN (LeCun, 1998) y el Pacto de Toledo (1995) para estabilidad financiera de sistemas, junto con aumentos en capacidad computacional, impulsaron aplicaciones prácticas. Desde 2012, modelos como Transformers han revolucionado áreas como procesamiento de lenguaje natural (Data Science Academy, 2025).

En 2006, Geoffrey Hinton contribuyó al desarrollo de las Deep Belief Networks (DBN), que utilizan preentrenamiento no supervisado para facilitar el entrenamiento de redes neuronales profundas. La introducción de las DBN marcó un hito en el avance del aprendizaje profundo, impulsando su aplicación en diversas áreas de la inteligencia artificial (Hinton y Osindero, 2006).

En 2012, Geoffrey Hinton y su equipo introdujeron AlexNet, un modelo de clasificación de imágenes que transformó el aprendizaje profundo. Este evento clave demostró las fortalezas de las arquitecturas de redes neuronales convolucionales (CNN) y sus amplias aplicaciones (Kingler, 2024).

Actualmente, las redes neuronales han alcanzado un nivel avanzado de sofisticación, impulsadas por modelos de lenguaje grande (LLM) como GPT-4, que mejoran la comprensión y generación de texto. Las redes neuronales convolucionales (CNN) dominan la visión por computadora, mientras que las redes generativas, como las GAN, producen contenido original de alta calidad. La transferencia de aprendizaje se ha vuelto estándar, facilitando la adaptación a nuevas tareas con pocos datos. Además, se presta atención a la ética y la mitigación de sesgos en la IA, y las redes neuronales están revolucionando el diagnóstico médico y la personalización de tratamientos. La integración de datos multimodales y el avance del hardware especializado continúan acelerando el desarrollo y la implementación de estas tecnologías en diversas industrias.

#### Timeline de Desarrollo de las Redes Neuronales

| Año | Hito |
|-----|------|
| 1943 | **Modelo McCulloch-Pitts**: Warren McCulloch y Walter Pitts proponen un modelo de neuronas artificiales usando circuitos eléctricos, sentando las bases para las redes neuronales |
| 1949 | **Aprendizaje Hebbiano**: Donald Hebb introduce el concepto de que las vías neuronales se fortalecen con la activación repetida, influyendo en modelos de aprendizaje posteriores |
| 1958 | **Desarrollo del Perceptrón**: Frank Rosenblatt desarrolla el perceptrón, una red neuronal temprana capaz de aprender de datos, limitada a tareas linealmente separables |
| 1969 | **Publicación de "Perceptrons"**: Minsky y Papert destacan las limitaciones de los perceptrones, particularmente su incapacidad para resolver problemas no lineales, llevando a una disminución del interés en las redes neuronales |
| 1970s-1980s | **Invierno de la IA**: Un período de reducción de financiamiento e investigación en IA y redes neuronales debido a las limitaciones destacadas por Minsky y Papert |
| 1986 | **Redescubrimiento de la Retropropagación**: Investigadores como Paul Werbos y David Rumelhart reviven el interés en las redes neuronales con la introducción de la retropropagación para entrenar redes multicapa |
| 1989 | **Redes Neuronales Convolucionales (CNNs)**: Yann LeCun introduce las CNNs, mejorando las capacidades de reconocimiento de imágenes y demostrando aplicaciones prácticas del aprendizaje profundo |
| 2006 | **Resurgimiento del Aprendizaje Profundo**: Geoffrey Hinton y otros introducen las redes de creencia profunda, marcando un resurgimiento en la investigación del aprendizaje profundo |
| 2012 | **Avance de AlexNet**: AlexNet de Alex Krizhevsky gana la competencia ImageNet, mostrando el poder del aprendizaje profundo en la clasificación de imágenes |
| 2020s | **Arquitecturas Transformadoras**: La emergencia de arquitecturas transformer revoluciona el procesamiento de lenguaje natural y otros campos, avanzando aún más las capacidades de IA |

*Nota: Referenciado por Codewave (2024)*

### 3.2 Principales arquitecturas: MLP, CNN, RNN



### 3.3 Algoritmos de entrenamiento: backpropagation, optimizadores



---

## Diseño e Implementación



---

## Arquitectura del Sistema



---

## Metodología de Desarrollo



---

## Resultados y Análisis



---

## Conclusiones



---

## Bibliografía

Data Science Academy. (2025). Capítulo 10 – As Principais Arquiteturas de Redes Neurais. Deep Learning Book. Recuperado de https://www.deeplearningbook.com.br/as-principais-arquiteturas-de-redes-neurais/

CodeWave. (2024). History and Development of Neural Networks in AI. Recuperado de https://codewave.com/insights/development-of-neural-networks-history/

Milner, Peter. (2003). A Brief History of the Hebbian Learning Rule. Canadian Psychology/Psychologie canadienne. 44. 5-9. https://doi.org/10.1037/h0085817

Rumelhart, David., Hinton, Geoffrey. & Williams, Ronald. (1986). Learning representations by back-propagating errors. Nature 323, 533–536. https://doi.org/10.1038/323533a0

Liu, Yuxi. (2024). The Perceptron Controversy. Yuxi on the Wired. Recuperado de: https://yuxi-liu-wired.github.io/essays/posts/perceptron-controversy/

Schaetti, Nils (2018) A Short history of Artificial Intelligence and Neural Network. LinkedIn. Recuperado de https://www.linkedin.com/pulse/short-history-artificial-intelligence-neural-network-nils

Hinton Geoffrey E., Osindero Simon. A fast learning algorithm for deep belief nets. Recuperado de https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

Kingler, Nico. (2024). AlexNet: A Revolutionary Deep Learning Architecture. Viso.ai Recuperado de https://viso.ai/deep-learning/alexnet/

Haykin, S. (2009). Neural Networks and Learning Machines. Prentice Hall. Recuperado de https://dai.fmph.uniba.sk/courses/NN/haykin.neural-networks.3ed.2009.pdf

---
