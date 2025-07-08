#include <iostream>
#include <locale>
#include <windows.h>
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/agent/PongAgent.h"
#include "pong_ascii.h"

using namespace utec::pong;

// Función para configurar UTF-8 en Windows
void setupUTF8() {
    // Configurar la consola para UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    // Configurar locale para UTF-8
    try {
        std::locale::global(std::locale(""));
        std::wcout.imbue(std::locale());
        std::cout.imbue(std::locale());
    } catch (const std::exception& e) {
        // Si falla el locale, usar configuración básica
        std::cout << "Advertencia: No se pudo configurar el locale UTF-8 completo.\n";
    }
    
    // Verificar que UTF-8 esté funcionando
    std::cout << "Configuración UTF-8 aplicada.\n";
}

int main() {
    // Configurar UTF-8 al inicio
    setupUTF8();
    
    EnvGym env(80, 24);          // ancho x alto
    PongAgent<float> agent;

    // Entrenamiento por política aleatoria (supervisado)
    // CONFIGURACIÓN ULTRA: 10,000 episodios, lotes de 10,000, máximo 5,000,000 muestras
    std::cout << "Iniciando entrenamiento ULTRA con 5M muestras...\n";
    agent.learnOnPolicy(env, 10000, 10000, 5000000);

    // Visualización en consola
    std::cout << "Iniciando simulación visual...\n";
    run_ascii_simulation(agent, 300, 1000); // pasos máximos y delay en ms

    return 0;
}
