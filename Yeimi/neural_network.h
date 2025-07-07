#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include <memory>
#include <vector>
#include <algorithm>

namespace utec::neural_network {
template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers;

    Tensor<T,2> get_batch(const Tensor<T,2>& data, size_t start, size_t end) const {
        std::array<size_t, 2> batch_shape = {end - start, data.shape()[1]};
        Tensor<T,2> batch(batch_shape);

        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < data.shape()[1]; ++j) {
                batch(i-start, j) = data(i, j);
            }
        }
        return batch;
    }

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    template <template <typename> class LossType, template <typename> class OptimizerType = Adam>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, const size_t epochs, const size_t batch_size, T learning_rate) {

        const size_t num_samples = X.shape()[0];
        const size_t num_batches = (num_samples + batch_size - 1) / batch_size;
        OptimizerType<T> optimizer(learning_rate);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start = batch * batch_size;
                size_t end = std::min(start + batch_size, num_samples);

                auto X_batch = get_batch(X, start, end);
                auto Y_batch = get_batch(Y, start, end);

                Tensor<T,2> out = X_batch;
                for (auto& layer : layers) {
                    out = layer->forward(out);
                }

                LossType<T> loss_fn(out, Y_batch);
                auto grad = loss_fn.loss_gradient();

                for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
                    grad = layers[i]->backward(grad);
                }

                for (auto& layer : layers) {
                    layer->update_params(optimizer);
                }
            }
        }
    }

    Tensor<T,2> predict(const Tensor<T,2>& X) {
        Tensor<T,2> output = X;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }
};

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H