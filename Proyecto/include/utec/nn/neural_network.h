#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include "interfaces.h"
#include "dense.h"
#include "optimizer.h"

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.emplace_back(std::move(layer));
        }

        template <template <typename...> class LossType, template <typename...> class OptimizerType = SGD>
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
                   const size_t epochs, const size_t batch_size, T learning_rate) {

            assert(X.shape()[0] == Y.shape()[0]);

            OptimizerType<T> optimizer(learning_rate);

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                // Forward
                Tensor<T,2> prediction = X;
                for (auto& layer : layers_)
                    prediction = layer->forward(prediction);

                // Loss
                LossType<T> loss_fn(prediction, Y);
                T loss = loss_fn.loss();

                // Backward
                Tensor<T,2> grad = loss_fn.loss_gradient();
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                    grad = (*it)->backward(grad);

                // Update params
                for (auto& layer : layers_)
                    layer->update_params(optimizer);
            }
        }

        Tensor<T,2> predict(const Tensor<T,2>& X) {
            Tensor<T,2> out = X;
            for (auto& layer : layers_)
                out = layer->forward(out);
            return out;
        }


    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
    };

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H