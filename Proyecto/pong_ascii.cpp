#include "pong_ascii.h"
#include "../include/utec/agent/EnvGym.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <windows.h>
#include <locale>

namespace utec::pong {

template <>
void run_ascii_simulation<float>(PongAgent<float>& agent, int max_steps, int delay_ms) {
    // Configurar UTF-8 para la simulación
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    const int max_lives = 3;
    int lives = max_lives;
    int score = 0;
    int games_played = 0;

    EnvGym env(80, 24);
    float reward;
    bool done = false;
    int steps = 0;

    auto state = env.reset();

    while (lives > 0) {
        int action = agent.act(state);
        state = env.step(action, reward, done);

        if (reward > 0) score += 1;

        system("cls"); // clear screen (use "clear" on Unix)

        for (int y = 0; y < 24; ++y) {
            for (int x = 0; x < 80; ++x) {
                if ((int)state.ball_x == x && (int)state.ball_y == y)
                    std::cout << "O";  // Bola
                else if (x == 0 &&
                         (y == (int)state.paddle_y ||
                          y == (int)state.paddle_y - 1 ||
                          y == (int)state.paddle_y + 1))
                    std::cout << "|";  // Paleta de 3 bloques
                else if (x == 79)  // Borde derecho
                    std::cout << "|";
                else if (y == 0 || y == 23)  // Bordes superior e inferior
                    std::cout << "-";
                else
                    std::cout << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Step: " << steps++ << " | Score: " << score << " | Lives: " << lives << " | Game: " << games_played + 1 << " | Ball: (" << (int)state.ball_x << "," << (int)state.ball_y << ")" << " | Paddle: " << (int)state.paddle_y << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

        if (done) {
            lives--;
            games_played++;
            state = env.reset();
            done = false;
            steps = 0;
            std::cout << "Perdió una vida. Reiniciando...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }

        if (steps >= max_steps) {
            state = env.reset();
            games_played++;
            std::cout << "Límite de pasos alcanzado. Reiniciando episodio...\n";
            steps = 0;
        }
    }

    std::cout << "\n¡Juego terminado! Puntaje final: " << score << " en " << games_played << " partidas.\n";
}

} // namespace utec::pong
