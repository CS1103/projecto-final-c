#include "pong_ascii.h"
#include "nn/neural_network.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

const int WIDTH = 40;
const int HEIGHT = 20;

void draw_screen(float ball_x, float ball_y, float paddle_y) {
    std::vector<std::string> screen(HEIGHT, std::string(WIDTH, ' '));

    // Dibujar bola
    int bx = static_cast<int>(ball_x);
    int by = static_cast<int>(ball_y);
    if (bx >= 0 && bx < WIDTH && by >= 0 && by < HEIGHT)
        screen[by][bx] = 'O';

    // Dibujar paleta (barra vertical)
    for (int i = -1; i <= 1; ++i) {
        int py = static_cast<int>(paddle_y) + i;
        if (py >= 0 && py < HEIGHT)
            screen[py][WIDTH - 2] = '|';
    }

    // Limpiar consola (ANSI escape)
    std::cout << "\033[2J\033[H";
    for (const auto& row : screen) {
        std::cout << row << "\n";
    }
}

void run_ascii_simulation(utec::neural_network::NeuralNetwork<float>& model) {
    float ball_x = WIDTH / 2.0f;
    float ball_y = HEIGHT / 2.0f;
    float dx = -1.0f, dy = 0.5f;
    float paddle_y = HEIGHT / 2.0f;

    while (true) {
        // -------- Convertir entrada a Tensor --------
        std::vector<float> input = { ball_x, ball_y, dx, dy, paddle_y };
        utec::algebra::Tensor<float, 2> input_tensor(1, input.size());
        for (int i = 0; i < input.size(); ++i)
            input_tensor(0, i) = input[i];

        // -------- Ejecutar predicción --------
        auto output_tensor = model.predict(input_tensor);

        // -------- Obtener acción --------
        std::vector<float> output_row(output_tensor.shape()[1]);
        for (size_t j = 0; j < output_tensor.shape()[1]; ++j)
            output_row[j] = output_tensor(0, j);

        int action = std::distance(output_row.begin(), std::max_element(output_row.begin(), output_row.end()));

        // -------- Mover paleta --------
        if (action == 1 && paddle_y > 1) paddle_y -= 1;
        if (action == 2 && paddle_y < HEIGHT - 2) paddle_y += 1;

        // -------- Mover bola --------
        ball_x += dx;
        ball_y += dy;

        // Rebote vertical
        if (ball_y <= 0 || ball_y >= HEIGHT - 1) dy *= -1;

        // Rebote en la paleta (lado derecho)
        if (ball_x >= WIDTH - 3 &&
            ball_y >= paddle_y - 1 &&
            ball_y <= paddle_y + 1) {
            dx *= -1;
        }

        // Si la bola sale del campo
        if (ball_x < 0 || ball_x >= WIDTH) {
            std::cout << "\nPUNTO. Reiniciando bola...\n";
            ball_x = WIDTH / 2.0f;
            ball_y = HEIGHT / 2.0f;
            dx = (rand() % 2 == 0) ? -1.0f : 1.0f;
            dy = ((rand() % 3) - 1) * 0.5f;
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }

        draw_screen(ball_x, ball_y, paddle_y);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
}
