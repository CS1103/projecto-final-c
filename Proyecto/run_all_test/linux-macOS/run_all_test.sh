#!/bin/bash

echo "============================"
echo "Ejecutando test_dense_layer"
echo "============================"
./cmake-build-debug/test_dense_layer

echo
echo "====================="
echo "Ejecutando test_ReLU"
echo "====================="
./cmake-build-debug/test_relu

echo
echo "=============================="
echo "Ejecutando test_convergence"
echo "=============================="
./cmake-build-debug/test_convergence

# Dar permisos de ejecución
chmod +x run_all_tests.sh

# ¿Cómo usarlo?
# 1. Navega al directorio donde se encuentra este script.
# 2. Asegúrate de que los ejecutables de prueba están en el directorio cmake-build-debug.
# 3. Ejecuta el script con el siguiente comando:
# ./run_all_tests.sh
