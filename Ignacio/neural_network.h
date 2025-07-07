//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#include <nn_loss.h>
#include <nn_optimizer.h>
#include <nn_interfaces.h>

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        std::unique_ptr<IOptimizer<T>> optimizer;

        public:

        NeuralNetwork() = default;

        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.push_back(std::move(layer));
        }

        void set_optimizer(std::unique_ptr<IOptimizer<T>> opt) {
            optimizer = std::move(opt);
        }

        Tensor<T,2> predict(const Tensor<T,2>& X) {
            return forward(X);
        }

        Tensor<T,2> forward(const Tensor<T,2>& X) {
            Tensor<T,2> output = X;
            for (auto& layer : layers) {
                output = layer->forward(output);
            }
            return output;
        }

        void backward(const Tensor<T, 2>& grad) {
            Tensor<T, 2> grad_input = grad;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad_input = (*it)->backward(grad_input);
            }
        }

        void optimize() {
            if (!optimizer) return;
            for (auto& layer : layers) {
                layer->update_params(*optimizer);
            }
            optimizer->step();
        }

        template<template<typename> class LossType = MSELoss,
             template<typename> class OptimizerType = SGD>
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
               size_t epochs, size_t batch_size, T learning_rate) {

            optimizer = std::make_unique<OptimizerType<T>>(learning_rate);

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                // Forward pass
                auto output = forward(X);
                LossType<T> loss(output, Y);

                backward(loss.loss_gradient());
                optimize();


            }
        }
    };
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
