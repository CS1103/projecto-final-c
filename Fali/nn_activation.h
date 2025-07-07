#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H


#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T>
class ReLU final: public ILayer<T> {
private:
    Tensor<T,2> cached_input_; //guarda la entrada para el backward
public:
    Tensor<T,2> forward(const Tensor<T,2>& z) override {
        cached_input_ = z; // guarda la entrada
        Tensor<T,2> output = z;

        for (size_t i = 0; i < output.shape()[0]; ++i) {
            for (size_t j = 0; j < output.shape()[1]; ++j) {
                output(i, j) = std::max(T(0), z(i, j));
            }
        }
        return output;
    }

    Tensor<T,2> backward(const Tensor<T,2>& gradients) override {
        Tensor<T,2> grad_input = gradients;

        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                grad_input(i, j) = (cached_input_(i, j) > T(0)) ? gradients(i, j) : T(0);
            }
        }
        return grad_input;
    }
};

template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    Tensor<T, 2> cached_output_;  // para usarlo en el backward
public:
    Tensor<T, 2> forward(const Tensor<T, 2>& z) override {
        cached_output_ = z;
        Tensor<T, 2> result = z;

        for (size_t i = 0; i < z.shape()[0]; ++i) {
            for (size_t j = 0; j < z.shape()[1]; ++j) {
                T val = z(i, j);
                result(i, j) = T(1) / (T(1) + std::exp(-val));
            }
        }

        cached_output_ = result;
        return result;
    }

    Tensor<T, 2> backward(const Tensor<T, 2>& gradients) override {
        Tensor<T, 2> grad_input = gradients;

        for (size_t i = 0; i < gradients.shape()[0]; ++i) {
            for (size_t j = 0; j < gradients.shape()[1]; ++j) {
                T sigmoid_val = cached_output_(i, j);
                grad_input(i, j) = gradients(i, j) * sigmoid_val * (T(1) - sigmoid_val);
            }
        }

        return grad_input;
    }
};




}

//funcion para que funcione el test 2 de la pregunta 1
template<typename T, size_t N, typename F>
utec::algebra::Tensor<T, N> apply(const utec::algebra::Tensor<T, N>& input, F func) {
    utec::algebra::Tensor<T, N> result = input;
    for (size_t i = 0; i < input.shape()[0]; ++i)
        for (size_t j = 0; j < input.shape()[1]; ++j)
            result(i, j) = func(input(i, j));
    return result;
}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
