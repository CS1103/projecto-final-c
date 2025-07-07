#include <iostream>
#include <fstream>
#include <vector>
#include "pong_ascii.h"
#include "utils.h"
#include "nn/neural_network.h"
#include "nn/dense.h"
#include "nn/activation.h"
#include "nn/loss.h"
#include "nn/optimizer.h"

int main() {
    // Cargar dataset
    std::vector<std::vector<float>> X, Y;
    load_dataset("dataset.csv", X, Y);

    // Convertir a Tensor
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    size_t n_outputs = Y[0].size();

    utec::neural_network::Tensor<float,2> X_tensor(n_samples, n_features);
    utec::neural_network::Tensor<float,2> Y_tensor(n_samples, n_outputs);

    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            X_tensor(i, j) = X[i][j];

    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_outputs; ++j)
            Y_tensor(i, j) = Y[i][j];

    // Crear red neuronal
    utec::neural_network::NeuralNetwork<float> model;
    model.add_layer(std::make_unique<utec::neural_network::Dense<float>>(5, 16,
        [](auto& w) { w.fill(0.01f); }, [](auto& b) { b.fill(0.0f); }));
    model.add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    model.add_layer(std::make_unique<utec::neural_network::Dense<float>>(16, 3,
        [](auto& w) { w.fill(0.01f); }, [](auto& b) { b.fill(0.0f); }));

    // Entrenar red usando train
    model.train<utec::neural_network::MSELoss, utec::neural_network::SGD>(
        X_tensor, Y_tensor, 50, 32, 0.01f // epochs, batch_size, learning_rate
    );

    // Ejecutar simulación Pong
    run_ascii_simulation(model);

    return 0;
}