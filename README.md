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

  * **Varela Villarreal, Yeimi Adelmar** – 202410586
  * **Guerrero Gutierrez, Nayeli Belén** – 202410790 
  * **Medina Patrick** – (completar código)
  * **Araoz Arana, Fali Ferdinand** – 202410721 
  * **Vidal Garcia Ignacio** – (completar código) 
  * **Bousssus Huerta Guillaume** – 202410017

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
Las redes neuronales artificiales (RNA) constituyen la base del aprendizaje profundo y representan uno de los avances más significativos en redes neuronales. Inspiradas en el funcionamiento del cerebro humano, estas estructuras computacionales permiten resolver problemas complejos que los algoritmos tradicionales no pueden abordar eficientemente. Este informe explora sus fundamentos teóricos, evolución histórica, arquitecturas principales y algoritmos de entrenamiento, proporcionando una base sólida para implementaciones prácticas.

##### Orígenes y Desarrollo Histórico

Dado el nacimiento de la informática moderna con la invención de la máquina de Turing en los 40s, la curiosidad por determinar lo que era computable o no eran la base de motivación de muchos autores por querer conocer la supremacía de la tecnología sobre los humanos. De ahí el test de Turing. Ambivalentemente, dos grandes como el neurólogo Warren McCulloch y el matemático Walter Pitts, motivados por la fuente de inteligencia humana, establecieron la primera arquitectura matemática de una neurona artificial. El concepto surgió dado este modelo.

Donald Hebb (1949) conocido también como el padre de la neuropsicología, dio a conocer el concepto abstracto de la "mente" con funciones cerebrales fisiológicas y biológicas específicas. Denominó el aprendizaje Hebbiano, definiendo que las neuronas forman redes y almacenan información en forma de recuerdos. "Las neuronas que se disparan juntas, se conectan" (Milner, 2003 p.1).

Frank Rosenblatt (1958) desarrolló el perceptrón, la primera red neuronal inspirado en el ojo de una mosca. Su importancia radica en ser un modelo fundamental para la clasificación de datos. Fue limitado, porque solo era efectivo para funciones lineales; esto fue demostrado por Marvin Minsky y Seymour Papert (1969) generando en la comunidad de conocedores una desilusión que terminó frenando la investigación sobre redes neuronales. Este tiempo se conoció como el "Invierno de la IA" que duró varias décadas entre finales de los 70 y finales de los 80 (Liu, 2024).

En 1986, David Rumelhart, Geoffrey Hinton y Ronald Williams publican un artículo que presenta el algoritmo de back-propagation. Con ello las investigaciones resurgieron. Este procedimiento permite el entrenamiento eficiente de redes neuronales multicapa al ajustar los pesos de las conexiones para minimizar la diferencia entre la salida real y la salida deseada (Rumelhart et al., 1986).

Avances como las redes convolucionales denominadas como CNN (LeCun, 1998) y el Pacto de Toledo (1995) para estabilidad financiera de sistemas, junto con aumentos en capacidad computacional, impulsaron aplicaciones prácticas. Desde 2012, modelos como Transformers han revolucionado áreas como procesamiento de lenguaje natural (Data Science Academy, 2025).

En 2006, Geoffrey Hinton contribuyó al desarrollo de las Deep Belief Networks (DBN), que utilizan preentrenamiento no supervisado para facilitar el entrenamiento de redes neuronales profundas. La introducción de las DBN marcó un hito en el avance del aprendizaje profundo, impulsando su aplicación en diversas áreas de la redes neuronales (Hinton y Osindero, 2006).

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

Después de revisar la historia de las redes neuronales, es relevante mencionar las 3 arquitecturas de estas:

* MLP (Multilayer Perceptron): Una MLP es una red neuronal compuesta por capas de neuronas completamente conectadas. Rumelhart, Hinton y Williams, (1986) “Las redes multicapa permite representar funciones no lineales mediante la combinación de múltiples capas con funciones de activación no lineales”. Esto quiere decir que una MLP puede aproximar cualquier función continua con la cantidad ideal de neuronas y capas.
  Esta estructura cuenta con capas de entrada, capas ocultas y una capa de salida.

* CNN (Convolutional Neural Network): Las CNN están diseñadas para procesar datos como imágenes. A diferencia de las MLP, usan capas que comparten pesos y se enfocan en regiones locales del espacio de entrada. LeCun et al., (1998) dicen que “Las convoluciones permiten a la red detectar características espaciales locales de manera eficiente y con menos parámetros”. Esta estructura cuenta con capas de convolución, capas de activación, capas de pooling y capas totalmente conectadas.

* RNN (Recurrent Neural Network): Las RNN son redes neuronales utilizadas para modelar datos donde el orden importa, como en texto, audio o series temporales. Incorporan una memoria interna que permite tener en cuenta entradas pasadas. Esta arquitectura cuenta con capas recurrentes, capas GRU (Gated Recurrent Unit), y capas LSTM (Long Short Term Memory) “Las LSTM pueden recordar información durante largos períodos gracias a su estructura de memoria controlada por compuertas” (Hochreiter & Schmidhuber, 1997).



#### 1.3 Algoritmos de entrenamiento: backpropagation, optimizadores
Por último, es importante revisar los algoritmos de entrenamiento utilizados para alimentar las redes neuronales.

* Backpropagation: El algoritmo de backpropagation, formalizado por Rumelhart, Hinton y Williams en 1986 consiste en utilizar la regla de la cadena para calcular los gradientes de la función de pérdida con respecto a cada peso en la red, desde la capa de salida hacia la capa de entrada. Esta retropropagación permite ajustar los pesos para minimizar el error del modelo. Este es esencial para minimizar el error durante el aprendizaje supervisado.
* Optimizadores: En este caso, hablaremos sobre un optimizador específico: Adam. El optimizador Adam (Adaptive Moment Estimation), propuesto por Kingma y Ba en 2014, combina las ventajas de algoritmos anteriores: utiliza la media y varianza de gradientes, ajustando la tasa de aprendizaje para cada parámetro. Además, corrige el sesgo inicial en las estimaciones y requiere poco ajuste manual. Esto lo hace particularmente efectivo en redes profundas entrenadas con datos grandes y ruidosos

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
1. Correr el archivo train_pong.cpp para iniciar el entrenamiento del agente y guardar el modelo entrenado. Después de 10,000 épocas, el agente debería aprender a jugar Pong.



### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

    * Iteraciones: 10 000 épocas(50 por lote).
    * Tiempo total de entrenamiento: 1h20m.
    * Precisión final: 75.8%.

* **Configuración de exploración:**:
    * Epsilon inicial: 1
    * Decay de Epsilon: 0.9995
    * Epsilon mínimo: 0.1

* **Ventajas/Desventajas**:


  🟢 VENTAJAS
1. **Arquitectura Modular y Bien Estructurada**

   Separación clara de responsabilidades: Tensor, Neural Network, Agent, Environment
   Uso de templates: Permite flexibilidad de tipos de datos (float, double)
   Namespaces organizados: utec::algebra, utec::nn, utec::pong
2. **Implementación Completa de Tensor**

   Soporte multidimensional: Tensores N-dimensionales con strides
   Broadcasting: Operaciones entre tensores de diferentes formas
   Operadores sobrecargados: +, -, *, / para operaciones matemáticas
   Matrix multiplication: Soporte para productos matriciales 2D y 3D
3. **Red Neuronal Funcional**

   Arquitectura flexible: Capas densas, activaciones (ReLU, Sigmoid)
   Sistema de capas: Interfaz ILayer<T> para extensibilidad
   Optimizadores: Implementación de optimizadores de gradiente
4. **Agente de IA Robusto**

   Epsilon-greedy: Balance entre exploración y explotación
   Entrenamiento por lotes: Eficiencia en el aprendizaje
   Estrategia híbrida: Combina reglas heurísticas con aprendizaje
5. **Visualización y Debugging**

   Simulación ASCII: Visualización en consola del juego
   Métricas de progreso: Seguimiento del entrenamiento
   Configuración UTF-8: Soporte para caracteres especiales
   
🔴 DESVENTAJAS

1. **Limitaciones del Algoritmo**

   Sin backpropagation real: El entrenamiento no actualiza pesos
   Aprendizaje supervisado básico. No implementa Q-learning o Policy Gradient
   Estrategia heurística dominante. El modelo neuronal tiene poco impacto
2. **Problemas de Rendimiento**

   Cálculos secuenciales: No aprovecha paralelización
   
    Memoria ineficiente: Copias innecesarias de tensores
   
    Sin optimizaciones: No usa BLAS o SIMD
3. **Falta de Robustez**

   Manejo de errores básico: Excepciones simples

   Sin validación de entrada: No verifica parámetros

   Configuración hardcodeada: Parámetros fijos en el código

* **Mejoras futuras**:

    * Uso de BLAS para multiplicaciones.




### 5. Trabajo en equipo

| Tarea | Miembro            | Rol |
|-------|--------------------|-----|
| Investigación teórica | Patrick Medina     | Documentar bases teóricas e historia de NNs |
| Desarrollo de arquitectura | Ignacio Vidal      | Diseño UML y esquemas de clases |
| Implementación del modelo | Yeimi Varela       | Código C++ del framework neuronal |
| Pruebas y benchmarking | Nayeli Guerrero    | Generación de métricas y testing |
| Documentación y demo | Fali Araoz         | Tutorial y demostración del sistema |
| Análisis de rendimiento | Guillaume Bousssus | Evaluación y optimización de performance |

> *Distribución de responsabilidades del equipo de desarrollo.*

---

### 6. Conclusiones

El proyecto Pong con redes neuronales representa una implementación completa y ambiciosa de un sistema de aprendizaje automático desarrollado íntegramente en C++ moderno, sin depender de bibliotecas externas. A través de este trabajo se demuestra la aplicación práctica de conceptos fundamentales de machine learning, álgebra lineal computacional y programación orientada a objetos en el contexto de un juego clásico, integrando múltiples disciplinas de forma rigurosa y estructurada.

La arquitectura del sistema ha sido diseñada con un enfoque modular, separando claramente las responsabilidades entre los distintos componentes: el núcleo matemático de tensores, la implementación de redes neuronales, el agente de IA y el entorno de simulación. El sistema de tensores N-dimensionales constituye la base computacional del proyecto, proporcionando operaciones como broadcasting automático, multiplicación matricial optimizada y manipulación eficiente de estructuras de datos multidimensionales. Esta infraestructura matemática permite implementar algoritmos de aprendizaje sin depender de herramientas externas, lo cual refuerza el entendimiento profundo de los fundamentos del machine learning.

Sobre esta base se construye una red neuronal flexible y extensible mediante un sistema de capas diseñadas con el patrón de diseño Strategy, lo que facilita su modificación y evolución. Se incluyen capas densas completamente conectadas, funciones de activación como ReLU y Sigmoid, y una estructura preparada para incorporar optimizadores de gradiente en el futuro. El agente de redes neuronales utiliza una estrategia híbrida que combina reglas heurísticas iniciales con aprendizaje basado en experiencia, empleando una política de exploración epsilon-greedy para mejorar progresivamente su rendimiento a través del juego.

El entorno de simulación, inspirado en la filosofía de interfaces como OpenAI Gym, permite una interacción limpia y coherente entre el agente y el juego. Esto facilita tanto el proceso de entrenamiento como la evaluación del modelo. Para la visualización, se optó por una interfaz en consola basada en caracteres ASCII, lo que ofrece una manera sencilla pero efectiva de observar en tiempo real el comportamiento del agente sin necesidad de librerías gráficas.

A nivel técnico, el proyecto destaca por su uso avanzado de características modernas de C++, incluyendo templates genéricos, gestión automática de memoria mediante el principio RAII, y sobrecarga de operadores para operaciones matemáticas intuitivas. Esta aproximación demuestra no solo habilidades avanzadas en programación, sino también un conocimiento profundo del lenguaje y su aplicabilidad a sistemas de alto rendimiento.

No obstante, el sistema aún presenta ciertas limitaciones que abren oportunidades claras para desarrollos futuros. La implementación actual del algoritmo de backpropagation no actualiza correctamente los pesos de la red neuronal, lo que restringe la capacidad de aprendizaje del modelo. Asimismo, la configuración del entrenamiento está codificada de forma estática, lo que dificulta la experimentación con diferentes hiperparámetros. La interfaz de usuario limitada a consola podría ampliarse con herramientas gráficas para facilitar la visualización de métricas y la manipulación del entrenamiento. Por otro lado, la cobertura de pruebas automatizadas es parcial y podría ampliarse para abarcar casos límite y validar algoritmos críticos con mayor rigor.

En conclusión, el proyecto de Pong con redes neuronales constituye un logro técnico significativo que combina teoría y práctica de manera ejemplar. No solo cumple con su objetivo educativo de demostrar cómo se construye un sistema de machine learning desde cero, sino que también establece una base técnica sólida para futuras investigaciones o desarrollos profesionales en redes neuronales y programación en C++. Su enfoque riguroso, su arquitectura bien definida y su ejecución técnica detallada convierten a este proyecto en un excelente ejemplo de integración disciplinaria y una valiosa adición a cualquier portafolio profesional enfocado en redes neuronales y desarrollo de software de alto rendimiento.

---

### 7. Bibliografía

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
