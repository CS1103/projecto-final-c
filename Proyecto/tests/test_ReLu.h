#ifndef TEST_RELU_H
#define TEST_RELU_H

#include <iostream>
#include "../include/utec/nn/activation.h"
#include "../include/utec/algebra/tensor.h"

void test_relu() {
    using namespace utec::neural_network;

    std::cout << "==== Test ReLU ====\n";

    // Caso 1: Test básico
    try {
        ReLU<float> relu;
        utec::algebra::Tensor<float, 2> input(1, 3);
        input = {-1.0f, 0.0f, 2.0f};
        auto output = relu.forward(input);
        std::cout << "[PASS] ReLU básico -> " << output(0, 0) << " " << output(0, 1) << " " << output(0, 2) << "\n";
    } catch (...) {
        std::cout << "[FAIL] Error inesperado en caso ReLU básico\n";
    }

    // Caso 2: Test de backprop
    try {
        ReLU<float> relu;
        utec::algebra::Tensor<float, 2> input(1, 3);
        input = {-2.0f, 0.0f, 5.0f};
        relu.forward(input);

        utec::algebra::Tensor<float, 2> grad_output(1, 3);
        grad_output = {1.0f, 1.0f, 1.0f};

        auto grad_input = relu.backward(grad_output);
        std::cout << "[PASS] Backward -> " << grad_input(0, 0) << " " << grad_input(0, 1) << " " << grad_input(0, 2) << "\n";
    } catch (...) {
        std::cout << "[FAIL] Error inesperado en backpropagation de ReLU\n";
    }

    // Caso 3: Tensor mal definido
    try {
        ReLU<float> relu;
        utec::algebra::Tensor<float, 2> input;  // vacío
        auto output = relu.forward(input);
        std::cout << "[FAIL] No se lanzó error con tensor vacío\n";
    } catch (...) {
        std::cout << "[PASS] Se capturó error con tensor inválido\n";
    }

    // Caso 4: Error de compilación (simulado como comentario)
    // ReLU<float> relu;
    // Tensor<float, 1> input(3);  // Error: se espera Tensor<...,2>
    // auto output = relu.forward(input);
}

#endif
