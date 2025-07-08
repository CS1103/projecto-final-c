#ifndef ENV_GYM_H
#define ENV_GYM_H

#include "state.h"
#include <random>

namespace utec::pong {

    class EnvGym {
    private:
        State state_;
        int width_;
        int height_;
        std::mt19937 rng_;
        int step_count_;
        static const int MAX_STEPS_PER_EPISODE = 1000;

    public:
        EnvGym(int width = 40, int height = 20)
            : width_(width), height_(height), rng_(std::random_device{}()), step_count_(0) {
            reset();
        }

        State reset() {
            step_count_ = 0;
            std::uniform_real_distribution<float> dist_y(0.0f, height_);
            std::uniform_real_distribution<float> dist_dy(-0.5f, 0.5f);
            std::uniform_int_distribution<int> dir(0, 1);

            state_.ball_x = width_ / 2.0f;
            state_.ball_y = height_ / 2.0f;
            state_.dx = dir(rng_) == 0 ? -1.0f : 1.0f;
            state_.dy = dist_dy(rng_);
            state_.paddle_y = height_ / 2.0f;

            // Asegurar que la pelota tenga velocidad mínima
            if (std::abs(state_.dy) < 0.1f) {
                state_.dy = state_.dy >= 0 ? 0.2f : -0.2f;
            }

            return state_;
        }

        State step(int action, float& reward, bool& done) {
            step_count_++;
            
            // Verificar límite de pasos para evitar bucles infinitos
            if (step_count_ >= MAX_STEPS_PER_EPISODE) {
                reward = -1.0f;
                done = true;
                return state_;
            }

            // Acción: -1 (arriba), 0 (quieto), +1 (abajo)
            if (action == -1 && state_.paddle_y > 1)
                state_.paddle_y -= 1.0f;
            else if (action == 1 && state_.paddle_y < height_ - 2)
                state_.paddle_y += 1.0f;

            // Mover la pelota
            state_.ball_x += state_.dx;
            state_.ball_y += state_.dy;

            // Rebote en bordes superior e inferior
            if (state_.ball_y <= 0 || state_.ball_y >= height_ - 1) {
                state_.dy *= -1.0f;
                // Asegurar que la pelota no se salga de los límites
                if (state_.ball_y <= 0) state_.ball_y = 0.0f;
                if (state_.ball_y >= height_ - 1) state_.ball_y = height_ - 1.0f;
            }

            // Golpea la paleta (lado izquierdo)
            if (state_.ball_x <= 1 &&
                state_.ball_y >= state_.paddle_y - 1 &&
                state_.ball_y <= state_.paddle_y + 1) {
                state_.dx *= -1.0f;
                // Recompensa positiva por golpear la pelota
                reward = 2.0f;
                done = false;
            }
            // Falla - la pelota sale por la izquierda
            else if (state_.ball_x <= 0) {
                reward = -5.0f; // Penalización más fuerte por perder
                done = true;
            }
            // Falla - la pelota sale por la derecha
            else if (state_.ball_x >= width_) {
                reward = -5.0f; // Penalización más fuerte por perder
                done = true;
            }
            // La pelota se sale por arriba o abajo (caso extremo)
            else if (state_.ball_y < 0 || state_.ball_y >= height_) {
                reward = -5.0f; // Penalización más fuerte por perder
                done = true;
            }
            else {
                // Recompensa pequeña por mantener la pelota en juego
                // y penalización por estar lejos de la pelota cuando se acerca
                if (state_.ball_x < width_ / 2) {
                    float distance_to_ball = std::abs(state_.paddle_y - state_.ball_y);
                    if (distance_to_ball < 2.0f) {
                        reward = 0.1f; // Pequeña recompensa por estar cerca de la pelota
                    } else {
                        reward = -0.01f * distance_to_ball; // Penalización por estar lejos
                    }
                } else {
                    reward = 0.0f; // Recompensa neutra cuando la pelota está lejos
                }
                done = false;
            }

            return state_;
        }

        const State& get_state() const {
            return state_;
        }
    };

} // namespace utec::pong

#endif // ENV_GYM_H
