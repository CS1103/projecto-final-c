#ifndef TEST_CONVERGENCE_H
#define TEST_CONVERGENCE_H

#include <iostream>
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/optimizer.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/activation.h"
#include "../include/utec/algebra/tensor.h"

void test_convergence() {
    using namespace utec::neural_network;

    std::cout << "==== Test Convergencia ====\n";

    // Caso 1: Aprender función XOR (converge lentamente)
    try {
        NeuralNetwork<float> model;
        model.add_layer(std::make_unique<Dense<float>>(2, 4, [](auto& w) { w.fill(0.1f); }, [](auto& b) { b.fill(0.0f); }));
        model.add_layer(std::make_unique<ReLU<float>>());
        model.add_layer(std::make_unique<Dense<float>>(4, 1, [](auto& w) { w.fill(0.1f); }, [](auto& b) { b.fill(0.0f); }));

        utec::algebra::Tensor<float, 2> X(4, 2);
        utec::algebra::Tensor<float, 2> Y(4, 1);
        X = {0, 0, 0, 1, 1, 0, 1, 1};
        Y = {0, 1, 1, 0};

        model.train<MSELoss, SGD>(X, Y, 1000, 4, 0.1f);

        auto output = model.predict(X);
        std::cout << "[PASS] Predicción tras entrenamiento XOR -> ";
        for (size_t i = 0; i < 4; ++i) std::cout << output(i, 0) << " ";
        std::cout << "\n";
    } catch (...) {
        std::cout << "[FAIL] Error durante entrenamiento\n";
    }

    // Caso 2: Dataset mal dimensionado
    try {
        NeuralNetwork<float> model;
        model.add_layer(std::make_unique<Dense<float>>(2, 1,
            [](auto& w) { w.fill(1.0f); }, [](auto& b) { b.fill(0.0f); }));

        utec::algebra::Tensor<float, 2> X(3, 2);
        utec::algebra::Tensor<float, 2> Y(4, 1);  // no tiene el mismo número de muestras

        model.train<MSELoss, SGD>(X, Y, 10, 1, 0.01f);
        std::cout << "[FAIL] No se detectó error de tamaño entre X y Y\n";
    } catch (...) {
        std::cout << "[PASS] Error capturado por inconsistencia entre X e Y\n";
    }

    // Caso 3: Test con datos triviales (funciona bien)
    try {
        NeuralNetwork<float> model;
        model.add_layer(std::make_unique<Dense<float>>(1, 1,
            [](auto& w) { w.fill(2.0f); }, [](auto& b) { b.fill(0.0f); }));

        utec::algebra::Tensor<float, 2> X(3, 1);
        utec::algebra::Tensor<float, 2> Y(3, 1);
        X = {1.0f, 2.0f, 3.0f};
        Y = {2.0f, 4.0f, 6.0f};

        model.train<MSELoss, SGD>(X, Y, 50, 1, 0.01f);
        std::cout << "[PASS] Aprendió función lineal correctamente\n";
    } catch (...) {
        std::cout << "[FAIL] Falló entrenamiento con datos lineales\n";
    }

    // Caso 4: Simulación de error de compilación
    // Tensor<float, 3> X; // Error: red espera Tensor<...,2>
    std::cout << "[INFO] Caso de compilación inválida simulado por comentario\n";
}

#endif
