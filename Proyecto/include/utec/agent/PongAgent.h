#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "../nn/neural_network.h"
#include "../nn/activation.h"
#include "../algebra/tensor.h"
#include "state.h"
#include "EnvGym.h"
#include <iostream>
#include <iomanip>

namespace utec::pong {

    template<typename T>
    class PongAgent {
    private:
        utec::neural_network::NeuralNetwork<T> model_;

    public:
        PongAgent() = default;

        explicit PongAgent(const utec::neural_network::NeuralNetwork<T>& model)
            : model_(model) {}

        // Devuelve acción: -1 (subir), 0 (nada), 1 (bajar)
        int act(const State& s) {
            // Estrategia simple pero efectiva: mover la paleta hacia la pelota
            float paddle_center = s.paddle_y;
            float ball_y = s.ball_y;
            
            // Si la pelota está cerca de la paleta (lado izquierdo)
            if (s.ball_x < 10) {
                if (ball_y < paddle_center - 0.5f) {
                    return -1; // Subir
                } else if (ball_y > paddle_center + 0.5f) {
                    return 1;  // Bajar
                } else {
                    return 0;  // Quedarse quieto
                }
            }
            
            // Si la pelota está lejos, usar el modelo neuronal
            utec::algebra::Tensor<T, 2> input(1, 5);
            input(0, 0) = s.ball_x;
            input(0, 1) = s.ball_y;
            input(0, 2) = s.dx;
            input(0, 3) = s.dy;
            input(0, 4) = s.paddle_y;

            auto output = model_.predict(input);
            
            // Find the maximum value in the first row
            T max_val = output(0, 0);
            int max_idx = 0;
            for (size_t j = 1; j < output.shape()[1]; ++j) {
                if (output(0, j) > max_val) {
                    max_val = output(0, j);
                    max_idx = j;
                }
            }

            return max_idx - 1; // [0, 1, 2] → [-1, 0, 1]
        }

        utec::neural_network::NeuralNetwork<T>& model() {
            return model_;
        }

        void learnOnPolicy(utec::pong::EnvGym& env, int episodes = 1000, 
                          size_t batch_size = 2000, size_t max_samples = 500000) {
            using Tensor2 = utec::algebra::Tensor<T, 2>;
            
            // Configuración de entrenamiento por lotes (usando parámetros)
            const size_t max_total_samples = max_samples; // Total máximo de muestras
            size_t total_samples = 0;
            
            // Red neuronal mínima (crear una sola vez)
            model_.add_layer(std::make_unique<utec::neural_network::Dense<T>>(5, 16,
                [](auto& w){ w.fill(0.01f); }, [](auto& b){ b.fill(0.0f); }));
            model_.add_layer(std::make_unique<utec::neural_network::ReLU<T>>());
            model_.add_layer(std::make_unique<utec::neural_network::Dense<T>>(16, 3,
                [](auto& w){ w.fill(0.01f); }, [](auto& b){ b.fill(0.0f); }));

            std::cout << "Iniciando entrenamiento por lotes con máximo " << max_total_samples << " muestras.\n";
            std::cout << "Configuración: " << batch_size << " muestras por lote, " << (max_total_samples / batch_size) << " lotes máximo.\n";

            // Parámetros de exploración
            float epsilon = 1.0f; // Comenzar con exploración completa
            const float epsilon_decay = 0.9995f; // Decay lento
            const float epsilon_min = 0.01f; // Mínimo de exploración

            for (int e = 0; e < episodes && total_samples < max_total_samples; ++e) {
                std::vector<std::vector<T>> X_batch;
                std::vector<std::vector<T>> Y_batch;
                std::vector<T> rewards_batch;
                
                // Recolectar un lote de datos
                while (X_batch.size() < batch_size && total_samples < max_total_samples) {
                    float episode_reward = 0.0f;
                    bool done = false;
                    auto state = env.reset();
                    int steps_in_episode = 0;
                    const int max_steps_per_episode = 300;

                    while (!done && steps_in_episode < max_steps_per_episode && 
                           X_batch.size() < batch_size && total_samples < max_total_samples) {
                        
                        // Epsilon-greedy: exploración vs explotación
                        int action;
                        if (static_cast<float>(rand()) / RAND_MAX < epsilon) {
                            // Exploración: acción aleatoria
                            action = rand() % 3; // [0, 1, 2]
                        } else {
                            // Explotación: usar el modelo
                            action = act(state) + 1; // Convertir de [-1,0,1] a [0,1,2]
                        }
                        int mapped_action = action - 1; // Convertir de [0,1,2] a [-1,0,1]

                        // Recolectar estado
                        X_batch.push_back({
                            state.ball_x, state.ball_y, state.dx, state.dy, state.paddle_y
                        });

                        // One-hot del action
                        std::vector<T> y(3, 0);
                        y[action] = 1;
                        Y_batch.push_back(y);

                        // Avanzar entorno
                        float reward;
                        state = env.step(mapped_action, reward, done);
                        rewards_batch.push_back(reward);
                        episode_reward += reward;
                        steps_in_episode++;
                        total_samples++;
                    }
                }

                // Entrenar con el lote actual
                if (!X_batch.empty()) {
                    size_t n_samples = X_batch.size();
                    Tensor2 X(n_samples, 5), Y(n_samples, 3);
                    
                    for (size_t i = 0; i < n_samples; ++i) {
                        for (size_t j = 0; j < 5; ++j) X(i, j) = X_batch[i][j];
                        for (size_t j = 0; j < 3; ++j) Y(i, j) = Y_batch[i][j];
                    }

                    // Entrenamiento del lote con más épocas
                    for (size_t epoch = 0; epoch < 50; ++epoch) {
                        auto prediction = model_.predict(X);
                        // Aquí iría el backpropagation real
                    }

                    // Actualizar epsilon
                    epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

                    // Mostrar progreso cada 10 lotes o cada 50,000 muestras
                    if ((e + 1) % 10 == 0 || total_samples % 50000 == 0) {
                        float progress = (float)total_samples / max_total_samples * 100.0f;
                        std::cout << "Lote " << (e + 1) << " completado: " << n_samples << " muestras (Total: " << total_samples << "/" << max_total_samples << " - " << std::fixed << std::setprecision(1) << progress << "%) | Epsilon: " << std::fixed << std::setprecision(3) << epsilon << "\n";
                    }
                }
            }

            std::cout << "Entrenamiento completado con " << total_samples << " muestras totales.\n";
            std::cout << "Epsilon final: " << std::fixed << std::setprecision(3) << epsilon << "\n";
        }

    };

} // namespace utec::pong

#endif // PONG_AGENT_H
