#include <iostream>
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.h"
#include "pong_ascii.h"

using namespace utec::pong;

int main() {
    // Crear entorno y agente
    EnvGym env(80, 24);
    PongAgent<float> agent;

    // Entrenamiento por política aleatoria
    std::cout << "[INFO] Iniciando entrenamiento del agente...\n";
    agent.learnOnPolicy(env, 10000, 10000, 5000000); // 10,000 episodios, lotes de 10,000, máximo 5M muestras

    // Simulación visual
    std::cout << "[INFO] Iniciando simulación visual del Pong...\n";
    run_ascii_simulation(agent, 300, 1000);

    return 0;
}
