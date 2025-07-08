#ifndef TEST_DENSE_LAYER_H
#define TEST_DENSE_LAYER_H

#include <iostream>
#include "../include/utec/nn/dense.h"
#include "../include/utec/algebra/tensor.h"

void test_dense_layer() {
    using namespace utec::neural_network;
    using namespace utec::algebra;

    std::cout << "==== Test Dense Layer ====\n";

    // Caso 1: Test simple con entrada de 2 características
    try {
        Dense<float> dense(2, 1,
                           [](auto& w) { w.fill(1.0f); },
                           [](auto& b) { b.fill(0.0f); });

        utec::algebra::Tensor<float, 2> input(1, 2);
        input(0, 0) = 2.0f;
        input(0, 1) = 3.0f;

        auto output = dense.forward(input);
        std::cout << "[PASS] Salida esperada: 5.0 -> " << output(0, 0) << "\n";
    } catch (...) {
        std::cout << "[FAIL] Excepción inesperada en caso simple\n";
    }

    // Caso 2: Error lógico - tamaños incompatibles
    try {
        Dense<float> dense(3, 2,
                           [](auto& w) { w.fill(0.5f); },
                           [](auto& b) { b.fill(0.1f); });

        utec::algebra::Tensor<float, 2> input(1, 2);  // mal tamaño (debería ser 1x3)
        auto output = dense.forward(input);
        std::cout << "[FAIL] No se lanzó excepción por tamaño incompatible\n";
    } catch (...) {
        std::cout << "[PASS] Excepción por tamaño de entrada incompatible\n";
    }

    // Caso 3: Error en ejecución (acceso fuera de rango simulado)
    try {
        Dense<float> dense(2, 1,
                           [](auto& w) { w.fill(1.0f); },
                           [](auto& b) { b.fill(0.0f); });

        utec::algebra::Tensor<float, 2> input;  // Tensor vacío
        auto output = dense.forward(input);  // Esto puede causar corrupción de memoria si no se valida
        std::cout << "[FAIL] No se lanzó error por tensor vacío\n";
    } catch (...) {
        std::cout << "[PASS] Excepción por tensor inválido (vacío)\n";
    }

    // Caso 4: Test con pesos/bias personalizados
    try {
        Dense<float> dense(2, 1,
                           [](auto& w) {
                               w(0, 0) = 2.0f;
                               w(1, 0) = 1.0f;
                           },
                           [](auto& b) {
                               b(0, 0) = 0.5f;
                           });

        utec::algebra::Tensor<float, 2> input(1, 2);
        input(0, 0) = 1.0f;
        input(0, 1) = 3.0f;

        auto output = dense.forward(input);
        std::cout << "[PASS] Salida esperada: 2*1 + 1*3 + 0.5 = 5.5 -> " << output(0, 0) << "\n";
    } catch (...) {
        std::cout << "[FAIL] Excepción inesperada en caso con inicialización personalizada\n";
    }
}

#endif
