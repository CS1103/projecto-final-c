//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"

#include <vector>
#include <memory>

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.push_back(std::move(layer));
        }

        template<template<typename> class LossType,
                 template<typename> class OptimizerType = SGD>
        void train(const utec::algebra::Tensor<T,2>& X,
                   const utec::algebra::Tensor<T,2>& Y,
                   size_t epochs,
                   size_t batch_size,
                   T lr) {
            OptimizerType<T> opt(lr);
            for (size_t e = 0; e < epochs; ++e) {
                auto out = X;
                for (auto& layer : layers_) out = layer->forward(out);
                LossType<T> loss_fn(out, Y);
                auto grad = loss_fn.loss_gradient();
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                    grad = (*it)->backward(grad);
                for (auto& layer : layers_) layer->update_params(opt);
            }
        }

        utec::algebra::Tensor<T,2> predict(const utec::algebra::Tensor<T,2>& X) {
            auto out = X;
            for (auto& layer : layers_) out = layer->forward(out);
            return out;
        }

    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
