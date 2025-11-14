


# **Perceptron CIFAR-10**

Implementación básica de un perceptrón para clasificación utilizando la base de datos **CIFAR-10**, con soporte para ejecución acelerada en GPU (T4) mediante **Google Colab**.



## **Integrantes**

* **Sergio Paucar**
* **Renato Oscar Corrales Peña**
* **Samuel Iman**

---

##  **Descripción del proyecto**

Este proyecto implementa un **perceptrón simple** con entrada y salida básica para clasificar imágenes del dataset **CIFAR-10**.
El código está diseñado para ejecutarse en **Google Colab** aprovechando una GPU **NVIDIA T4** para mejorar el rendimiento.

---

##  **Instrucciones de ejecución en Google Colab**

### 1 Cambiar el entorno a GPU (T4)

En Colab:
**Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)**

---

### 2 Subir los 3 archivos del proyecto

Sube los archivos:

* `main.cu`
* `cifar10_loader.cpp`
* `cifar10_loader.h`

---

### 3 Verificar la GPU

Ejecuta en una celda:

```bash
!nvidia-smi
```

---

### 4 Descargar y preparar el dataset CIFAR-10

Ejecuta:

```bash
!mkdir -p data
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xvzf cifar-10-binary.tar.gz
!cp cifar-10-batches-bin/* data/
!ls data
```

---

### 5 Compilar y ejecutar el programa

Ejecuta:

```bash
!nvcc -O3 -use_fast_math -std=c++17 -arch=sm_75 main.cu cifar10_loader.cpp -o main
!./main
```

---


## 📝 **Notas**

* Asegúrate de estar usando una GPU compatible (T4 o superior).
* Si modificas el código fuente, compila de nuevo antes de ejecutar.

---


