cmake_minimum_required(VERSION 3.29.5)
project(Proyecto)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir carpeta con headers
include_directories(
        ${PROJECT_SOURCE_DIR}/nn
)

# Archivos fuente
add_executable(Proyecto
        main.cpp
        pong_ascii.cpp
)

# Uso de librerías externas
