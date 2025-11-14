# Perceptron_Cifar_10
## Integrantes 
  Sergio Paucar 
  Renato Oscar Corrales Peña
  Samuel iman
*Perceptrón de entrada y salida basico
*Se necesita usar Google Colab, una vez abierto cambiar el entrono de ejecución a gpru T4
*Subir los tres archivos a Google Colab
*Luego inicializar el GPU poniendo este comando en una celda
*!nvidia-smi
*Descargar la data CIFAR_10, ejecutando esta celda
*!mkdir -p data
*!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
*!tar -xvzf cifar-10-binary.tar.gz
*!cp cifar-10-batches-bin/* data/
*!ls data
*Para ejecutar el main poner esta celda de código
*!nvcc -O3 -use_fast_math -std=c++17 -arch=sm_75 main.cu cifar10_loader.cpp -o main
*!./main

