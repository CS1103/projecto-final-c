@echo off
echo ============================
echo Ejecutando test_dense_layer
echo ============================
.\cmake-build-debug\test_dense_layer.exe

echo ======================
echo Ejecutando test_ReLU
echo ======================
.\cmake-build-debug\test_relu.exe

echo ===============================
echo Ejecutando test_convergence
echo ===============================
.\cmake-build-debug\test_convergence.exe

pause

REM Dar permisos de ejecución al script:
REM icacls run_all_test.bat /grant %username%:F
REM Nota: Asegúrate de que el directorio cmake-build-debug contiene los ejecutables generados por CMake.
REM Para ejecutar el script, abrir una terminal de Windows y navegar al directorio donde se encuentra el script.
REM Luego, ejecutar el comando:
REM run_all_test.bat
