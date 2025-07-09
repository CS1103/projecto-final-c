# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++ para el juego Pong, desarrollando desde cero un framework completo de redes neuronales artificiales sin dependencias externas especializadas.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)

---

### Datos generales

* **Tema**: Redes Neuronales en AI aplicadas al juego Pong
* **Grupo**: `C--`
* **Integrantes**:

  * **Varela Villarreal, Yeimi Adelmar** – 
  * **Guerrero Gutierrez, Nayeli Belén** – 202410790 
  * **Medina Patrick** – (completar código)
  * **Araoz, Fali** – (completar código) 
  * **Ignacio** – (completar código) 
  * **Guillaume** – (completar código)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior / MSVC 2019+
2. **Dependencias**:

   * CMake 3.18+
   * C++20 estándar
   * Ninguna librería externa especializada (implementación propia)

3. **Instalación**:

   ```bash
   git clone git@github.com:CS1103/projecto-final-c.git
   cd projecto-final-c/Proyecto
   mkdir build && cd build
   cmake ..
   make
   ```

> *Comandos para compilación en Windows y Linux.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales aplicadas al aprendizaje automático.
* **Contenido desarrollado**:

  1. Historia y evolución de las NNs.
  2. Principales arquitecturas: MLP, CNN, RNN.
  3. Algoritmos de entrenamiento: backpropagation, optimizadores.

#### 1.1 Historia y evolución de las NNs

##### Introducción Teórica
Las redes neuronales artificiales (RNA) constituyen la base del aprendizaje profundo y representan uno de los avances más significativos en inteligencia artificial. Inspiradas en el funcionamiento del cerebro humano, estas estructuras computacionales permiten resolver problemas complejos que los algoritmos tradicionales no pueden abordar eficientemente. Este informe explora sus fundamentos teóricos, evolución histórica, arquitecturas principales y algoritmos de entrenamiento, proporcionando una base sólida para implementaciones prácticas.

##### Orígenes y Desarrollo Histórico

Dado el nacimiento de la informática moderna con la invención de la máquina de Turing en los 40s, la curiosidad por determinar lo que era computable o no eran la base de motivación de muchos autores por querer conocer la supremacía de la tecnología sobre los humanos. De ahí el test de Turing. Ambivalentemente, dos grandes como el neurólogo Warren McCulloch y el matemático Walter Pitts, motivados por la fuente de inteligencia humana, establecieron la primera arquitectura matemática de una neurona artificial. El concepto surgió dado este modelo.

Donald Hebb (1949) conocido también como el padre de la neuropsicología, dio a conocer el concepto abstracto de la "mente" con funciones cerebrales fisiológicas y biológicas específicas. Denominó el aprendizaje Hebbiano, definiendo que las neuronas forman redes y almacenan información en forma de recuerdos. "Las neuronas que se disparan juntas, se conectan" (Milner, 2003 p.1).

Frank Rosenblatt (1958) desarrolló el perceptrón, la primera red neuronal inspirado en el ojo de una mosca. Su importancia radica en ser un modelo fundamental para la clasificación de datos. Fue limitado, porque solo era efectivo para funciones lineales; esto fue demostrado por Marvin Minsky y Seymour Papert (1969) generando en la comunidad de conocedores una desilusión que terminó frenando la investigación sobre redes neuronales. Este tiempo se conoció como el "Invierno de la IA" que duró varias décadas entre finales de los 70 y finales de los 80 (Liu, 2024).

En 1986, David Rumelhart, Geoffrey Hinton y Ronald Williams publican un artículo que presenta el algoritmo de back-propagation. Con ello las investigaciones resurgieron. Este procedimiento permite el entrenamiento eficiente de redes neuronales multicapa al ajustar los pesos de las conexiones para minimizar la diferencia entre la salida real y la salida deseada (Rumelhart et al., 1986).

Avances como las redes convolucionales denominadas como CNN (LeCun, 1998) y el Pacto de Toledo (1995) para estabilidad financiera de sistemas, junto con aumentos en capacidad computacional, impulsaron aplicaciones prácticas. Desde 2012, modelos como Transformers han revolucionado áreas como procesamiento de lenguaje natural (Data Science Academy, 2025).

En 2006, Geoffrey Hinton contribuyó al desarrollo de las Deep Belief Networks (DBN), que utilizan preentrenamiento no supervisado para facilitar el entrenamiento de redes neuronales profundas. La introducción de las DBN marcó un hito en el avance del aprendizaje profundo, impulsando su aplicación en diversas áreas de la inteligencia artificial (Hinton y Osindero, 2006).

En 2012, Geoffrey Hinton y su equipo introdujeron AlexNet, un modelo de clasificación de imágenes que transformó el aprendizaje profundo. Este evento clave demostró las fortalezas de las arquitecturas de redes neuronales convolucionales (CNN) y sus amplias aplicaciones (Kingler, 2024).

Actualmente, las redes neuronales han alcanzado un nivel avanzado de sofisticación, impulsadas por modelos de lenguaje grande (LLM) como GPT-4, que mejoran la comprensión y generación de texto. Las redes neuronales convolucionales (CNN) dominan la visión por computadora, mientras que las redes generativas, como las GAN, producen contenido original de alta calidad. La transferencia de aprendizaje se ha vuelto estándar, facilitando la adaptación a nuevas tareas con pocos datos. Además, se presta atención a la ética y la mitigación de sesgos en la IA, y las redes neuronales están revolucionando el diagnóstico médico y la personalización de tratamientos. La integración de datos multimodales y el avance del hardware especializado continúan acelerando el desarrollo y la implementación de estas tecnologías en diversas industrias.

##### Timeline de Desarrollo de las Redes Neuronales

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

#### 1.2 Principales arquitecturas: MLP, CNN, RNN

*[Contenido a completar por los compañeros de equipo]*

#### 1.3 Algoritmos de entrenamiento: backpropagation, optimizadores

*[Contenido a completar por los compañeros de equipo]*



---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño implementados**: Factory para creación de capas, Strategy para optimizadores, Observer para monitoreo de entrenamiento.
* **Estructura de carpetas del proyecto**:

  ```
  Proyecto/
  ├── CMakeLists.txt
  ├── pong_ascii.cpp/h          # Visualización del juego
  ├── train_pong.cpp            # Entrenamiento principal
  ├── utils.h                   # Utilidades generales
  ├── include/utec/
  │   ├── algebra/
  │   │   └── tensor.h          # Sistema de tensores
  │   ├── nn/
  │   │   ├── neural_network.h  # Red neuronal principal
  │   │   ├── dense.h           # Capas densas
  │   │   ├── activation.h      # Funciones de activación
  │   │   ├── optimizer.h       # Optimizadores
  │   │   ├── loss.h            # Funciones de pérdida
  │   │   └── interfaces.h      # Interfaces polimórficas
  │   └── agent/
  │       ├── PongAgent.h       # Agente inteligente
  │       ├── EnvGym.h          # Entorno de simulación
  │       └── state.h           # Estado del juego
  └── tests/                    # Pruebas unitarias
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar el proyecto**:
  ```bash
  # Compilar el proyecto
  cd Proyecto/cmake-build-debug
  
  # Ejecutar entrenamiento
  ./train_pong.exe
  
  # Ejecutar simulación visual
  ./Proyecto.exe
  ```

* **Casos de prueba implementados**:
  * Test unitario de tensores multidimensionales
  * Test de capa densa con forward/backward pass
  * Test de función de activación ReLU
  * Test de convergencia del algoritmo Adam
  * Test de rendimiento del agente en Pong

> *Archivos de prueba ubicados en la carpeta `tests/`*

---

### 3. Ejecución



**Pasos para ejecutar:**



### 4. Análisis del rendimiento



### 5. Trabajo en equipo

| Tarea | Miembro | Rol |
|-------|---------|-----|
| Investigación teórica | | Documentar bases teóricas e historia de NNs |
| Desarrollo de arquitectura | | Diseño UML y esquemas de clases |
| Implementación del modelo |  | Código C++ del framework neuronal |
| Pruebas y benchmarking |  | Generación de métricas y testing |
| Documentación y demo |  | Tutorial y demostración del sistema |
| Análisis de rendimiento |  | Evaluación y optimización de performance |

> *Distribución de responsabilidades del equipo de desarrollo.*

---

### 6. Conclusiones



### 7. Bibliografía

> *Referencias bibliográficas en formato IEEE:*



---

### 7. Bibliografía

> *Referencias bibliográficas en formato IEEE:*

[1] Data Science Academy. (2025). "Capítulo 10 – As Principais Arquiteturas de Redes Neurais," *Deep Learning Book*. [Online]. Available: https://www.deeplearningbook.com.br/as-principais-arquiteturas-de-redes-neurais/

[2] CodeWave. (2024). "History and Development of Neural Networks in AI." [Online]. Available: https://codewave.com/insights/development-of-neural-networks-history/

[3] P. Milner, "A Brief History of the Hebbian Learning Rule," *Canadian Psychology/Psychologie canadienne*, vol. 44, pp. 5-9, 2003. DOI: 10.1037/h0085817

[4] D. Rumelhart, G. Hinton, and R. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533-536, 1986. DOI: 10.1038/323533a0

[5] Y. Liu, "The Perceptron Controversy," *Yuxi on the Wired*, 2024. [Online]. Available: https://yuxi-liu-wired.github.io/essays/posts/perceptron-controversy/

[6] N. Schaetti, "A Short history of Artificial Intelligence and Neural Network," *LinkedIn*, 2018. [Online]. Available: https://www.linkedin.com/pulse/short-history-artificial-intelligence-neural-network-nils

[7] G. E. Hinton and S. Osindero, "A fast learning algorithm for deep belief nets." [Online]. Available: https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

[8] N. Kingler, "AlexNet: A Revolutionary Deep Learning Architecture," *Viso.ai*, 2024. [Online]. Available: https://viso.ai/deep-learning/alexnet/

[9] S. Haykin, *Neural Networks and Learning Machines*, Prentice Hall, 2009. [Online]. Available: https://dai.fmph.uniba.sk/courses/NN/haykin.neural-networks.3ed.2009.pdf

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.

---
