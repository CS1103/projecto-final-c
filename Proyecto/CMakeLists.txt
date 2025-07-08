cmake_minimum_required(VERSION 3.20)
project(Proyecto)

set(CMAKE_CXX_STANDARD 20)

# Archivos fuente principales
add_executable(Proyecto
        main.cpp
        pong_ascii.cpp
        include/utec/agent/state.h
        include/utec/agent/EnvGym.h
        include/utec/agent/PongAgent.h
        train_pong.cpp
)

# Incluir encabezados
target_include_directories(Proyecto PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

# TESTS
add_executable(test_dense_layer tests/test_dense_layer.cpp)
target_include_directories(test_dense_layer PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable(test_relu tests/test_ReLu.cpp)
target_include_directories(test_relu PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable(test_convergence tests/test_convergence.cpp)
target_include_directories(test_convergence PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_executable(train_pong train_pong.cpp pong_ascii.cpp)
target_include_directories(train_pong PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)
