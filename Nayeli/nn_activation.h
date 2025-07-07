#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#include "tensor.h"
#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

    template<typename T, size_t DIMS>
    using Tensor = utec::algebra::Tensor<T, DIMS>;

    // Clase ReLU corregida
    template<typename T>
    class ReLU final : public ILayer<T> {
    private:
        Tensor<T,2> last_input;
    public:
        // Constructor por defecto corregido - usa constructor explícito con std::array
        ReLU() : last_input(std::array<size_t, 2>{1, 1}) {}

        Tensor<T,2> forward(const Tensor<T,2>& z) override {
            last_input = z;
            Tensor<T,2> result(z.shape());
            for (size_t i = 0; i < z.size(); ++i) {
                result.data[i] = (z.data[i] < 0) ? 0 : z.data[i];
            }
            return result;
        }

        Tensor<T,2> backward(const Tensor<T,2>& g) override {
            Tensor<T,2> grad(g.shape());
            for (size_t i = 0; i < g.size(); ++i) {
                grad.data[i] = (last_input.data[i] > 0) ? g.data[i] : 0;
            }
            return grad;
        }

        // Implementación vacía para funciones de activación (no tienen parámetros)
        void update_params(IOptimizer<T>& optimizer) override {
            // Las funciones de activación no tienen parámetros para actualizar
        }
    };

    // Clase Sigmoid corregida
    template<typename T>
    class Sigmoid final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T,2> last_output;
    public:
        // Constructor por defecto
        Sigmoid() : last_output(std::array<size_t, 2>{1, 1}) {}

        utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
            // FIX: Crear tensor con el shape correcto usando el constructor adecuado
            last_output = utec::algebra::Tensor<T,2>(z.shape());
            for (size_t i = 0; i < z.size(); ++i) {
                last_output.data[i] = 1.0 / (1.0 + std::exp(-z.data[i]));
            }
            return last_output;
        }

        utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& g) override {
            // FIX: Crear tensor con el shape correcto usando el constructor adecuado
            utec::algebra::Tensor<T,2> grad(g.shape());
            for (size_t i = 0; i < g.size(); ++i) {
                grad.data[i] = g.data[i] * last_output.data[i] * (1 - last_output.data[i]);
            }
            return grad;
        }

        // Implementación vacía para funciones de activación (no tienen parámetros)
        void update_params(IOptimizer<T>& optimizer) override {
            // Las funciones de activación no tienen parámetros para actualizar
        }
    };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H