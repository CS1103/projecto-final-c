#ifndef PONG_ASCII_H
#define PONG_ASCII_H
#pragma once
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/agent/PongAgent.h"

namespace utec::pong {

    template <typename T>
    void run_ascii_simulation(PongAgent<T>& agent, int max_steps = 300, int delay_ms = 1000);



    void run_ascii_simulation(utec::neural_network::NeuralNetwork<float>& model);
}
// namespace utec::pong

#endif
