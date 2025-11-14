// perceptron_rosenblatt_cuda.cu
// Compilar: nvcc -O3 perceptron_rosenblatt_cuda.cu -o perceptron_cuda
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>
#include "cifar10_loader.h" // Debe proporcionar loadCIFAR10Batch y CIFAR10Data

#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

// Kernel forward: Z = X * W + b
__global__ void forwardBatchKernel(const float *X, const float *W, const float *b, float *Z,
                                   int batch_size, int input_size, int num_classes)
{
    int sample = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample < batch_size && j < num_classes)
    {
        float sum = 0.0f;
        const float *x = X + sample * input_size;

        for (int k = 0; k < input_size; k++)
            sum += x[k] * W[k * num_classes + j];

        Z[sample * num_classes + j] = sum + b[j];
    }
}

// Kernel Rosenblatt update: cada hilo procesa 1 muestra y hace atomicAdd en W y b
__global__ void rosenblattUpdateKernel(float *W, float *b,
                                       const float *X, const int *Y_labels,
                                       const float *Z,
                                       int batch_size, int input_size, int num_classes,
                                       float learning_rate, int *d_update_count)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch_size) return;

    const float *z = Z + sample * num_classes;
    int pred = 0;
    float maxv = z[0];
    for (int j = 1; j < num_classes; ++j) {
        if (z[j] > maxv) { maxv = z[j]; pred = j; }
    }
    int true_class = Y_labels[sample];
    if (pred == true_class) return;

    const float *x = X + sample * input_size;
    float lr = learning_rate;

    for (int k = 0; k < input_size; ++k) {
        atomicAdd(&W[k * num_classes + true_class], lr * x[k]);
        atomicAdd(&W[k * num_classes + pred], -lr * x[k]);
    }
    atomicAdd(&b[true_class], lr);
    atomicAdd(&b[pred], -lr);
    atomicAdd(d_update_count, 1);
}

void save_checkpoint(const std::string &file, const std::vector<float> &W, const std::vector<float> &b) {
    std::ofstream ofs(file, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(W.data()), W.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(float));
    ofs.close();
}

int main() {
    std::cout << "Perceptron Rosenblatt (CUDA) - CIFAR-10\n";
    const int NUM_CLASSES = 10;
    const int INPUT_SIZE = 3072;
    const int EPOCHS = 500;
    float LEARNING_RATE = 0.001f;
    const int CHECKPOINT_INTERVAL = 10;

    // Cargar datos
    CIFAR10Data train_data;
    for (int i = 1; i <= 5; ++i) {
        std::string filename = "data/data_batch_" + std::to_string(i) + ".bin";
        CIFAR10Data batch = loadCIFAR10Batch(filename);
        train_data.images.insert(train_data.images.end(), batch.images.begin(), batch.images.end());
        train_data.labels.insert(train_data.labels.end(), batch.labels.begin(), batch.labels.end());
    }
    CIFAR10Data test_data = loadCIFAR10Batch("data/test_batch.bin");
    int n_train = (int)train_data.images.size();
    int n_test = (int)test_data.images.size();
    std::cout << "Train samples: " << n_train << "  Test samples: " << n_test << '\n';

    // Inicializar W y b
    std::vector<float> W(INPUT_SIZE * NUM_CLASSES);
    std::vector<float> b(NUM_CLASSES, 0.0f);
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto &w : W) w = dist(gen);

    // Device allocations
    float *d_W=nullptr, *d_b=nullptr, *d_X=nullptr, *d_Z=nullptr;
    int *d_Y=nullptr, *d_update_count=nullptr;

    cudaMalloc(&d_W, W.size() * sizeof(float));
    cudaMalloc(&d_b, b.size() * sizeof(float));
    cudaMalloc(&d_X, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_Z, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_Y, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_update_count, sizeof(int));

    cudaMemcpy(d_W, W.data(), W.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Host buffers for batches
    std::vector<float> X_batch(BATCH_SIZE * INPUT_SIZE);
    std::vector<int> Y_batch(BATCH_SIZE);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto ep_start = std::chrono::high_resolution_clock::now();
        int total_updates = 0;

        // Shuffle indices
        std::vector<int> idx(n_train);
        for (int i = 0; i < n_train; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), gen);

        int num_batches = (n_train + BATCH_SIZE - 1) / BATCH_SIZE;
        for (int bidx = 0; bidx < num_batches; ++bidx) {
            int start = bidx * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, n_train);
            int current_bs = end - start;

            // Fill host batch
            for (int i = 0; i < current_bs; ++i) {
                const auto &img = train_data.images[idx[start + i]];
                std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
                Y_batch[i] = train_data.labels[idx[start + i]];
            }
            if (current_bs < BATCH_SIZE) {
                std::fill(X_batch.begin() + current_bs * INPUT_SIZE, X_batch.end(), 0.0f);
                std::fill(Y_batch.begin() + current_bs, Y_batch.end(), 0);
            }

            // Copy to device
            cudaMemcpy(d_X, X_batch.data(), current_bs * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Y, Y_batch.data(), current_bs * sizeof(int), cudaMemcpyHostToDevice);

            // Forward kernel
            dim3 grid((NUM_CLASSES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, current_bs);
            forwardBatchKernel<<<grid, THREADS_PER_BLOCK>>>(d_X, d_W, d_b, d_Z, current_bs, INPUT_SIZE, NUM_CLASSES);

            // Reset counter
            cudaMemset(d_update_count, 0, sizeof(int));

            // Update kernel
            int blocks = (current_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            rosenblattUpdateKernel<<<blocks, THREADS_PER_BLOCK>>>(d_W, d_b, d_X, d_Y, d_Z, current_bs, INPUT_SIZE, NUM_CLASSES, LEARNING_RATE, d_update_count);

            cudaDeviceSynchronize();

            // Read updates
            int updates_batch = 0;
            cudaMemcpy(&updates_batch, d_update_count, sizeof(int), cudaMemcpyDeviceToHost);
            total_updates += updates_batch;
        } // end batches

        auto ep_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(ep_end - ep_start).count() / 1000.0;

        // Evaluate on test set (batch eval)
        int correct = 0;
        std::vector<float> Z_host(BATCH_SIZE * NUM_CLASSES);
        for (int tb = 0; tb < (n_test + BATCH_SIZE - 1) / BATCH_SIZE; ++tb) {
            int start = tb * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, n_test);
            int cur_bs = end - start;

            for (int i = 0; i < cur_bs; ++i) {
                const auto &img = test_data.images[start + i];
                std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
            }
            if (cur_bs < BATCH_SIZE) std::fill(X_batch.begin() + cur_bs * INPUT_SIZE, X_batch.end(), 0.0f);

            cudaMemcpy(d_X, X_batch.data(), cur_bs * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            dim3 gridEval((NUM_CLASSES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, cur_bs);
            forwardBatchKernel<<<gridEval, THREADS_PER_BLOCK>>>(d_X, d_W, d_b, d_Z, cur_bs, INPUT_SIZE, NUM_CLASSES);
            cudaDeviceSynchronize();
            cudaMemcpy(Z_host.data(), d_Z, cur_bs * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < cur_bs; ++i) {
                float *scores = Z_host.data() + i * NUM_CLASSES;
                int pred = std::distance(scores, std::max_element(scores, scores + NUM_CLASSES));
                if (pred == test_data.labels[start + i]) correct++;
            }
        }

        double acc = 100.0 * correct / n_test;
        std::cout << "Epoch " << (epoch+1) << "/" << EPOCHS << " | updates: " << total_updates << " | acc: " << acc << "% | time: " << epoch_time << "s\n";

        // Checkpoint
        if ((epoch + 1) % CHECKPOINT_INTERVAL == 0 || epoch == EPOCHS - 1) {
            cudaMemcpy(W.data(), d_W, W.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), d_b, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
            std::string fname = "cuda_checkpoint_epoch_" + std::to_string(epoch+1) + ".bin";
            save_checkpoint(fname, W, b);
            std::cout << " → Checkpoint saved: " << fname << '\n';
        }
    } // end epochs

    // ======================================================
// EVALUACIÓN FINAL DESPUÉS DE TODO EL ENTRENAMIENTO
// ======================================================
std::cout << "\n-----------------------------------------\n";
std::cout << "      Evaluando precisión FINAL...\n";
std::cout << "-----------------------------------------\n";

int final_correct = 0;
std::vector<float> Z_host_final(BATCH_SIZE * NUM_CLASSES);

for (int tb = 0; tb < (n_test + BATCH_SIZE - 1) / BATCH_SIZE; ++tb) {
    int start = tb * BATCH_SIZE;
    int end = std::min(start + BATCH_SIZE, n_test);
    int cur_bs = end - start;

    for (int i = 0; i < cur_bs; ++i) {
        const auto &img = test_data.images[start + i];
        std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
    }
    if (cur_bs < BATCH_SIZE)
        std::fill(X_batch.begin() + cur_bs * INPUT_SIZE, X_batch.end(), 0.0f);

    cudaMemcpy(d_X, X_batch.data(), cur_bs * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridFinal((NUM_CLASSES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, cur_bs);
    forwardBatchKernel<<<gridFinal, THREADS_PER_BLOCK>>>(d_X, d_W, d_b, d_Z,
                                                         cur_bs, INPUT_SIZE, NUM_CLASSES);
    cudaDeviceSynchronize();

    cudaMemcpy(Z_host_final.data(), d_Z, cur_bs * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < cur_bs; ++i) {
        float *scores = Z_host_final.data() + i * NUM_CLASSES;
        int pred = std::distance(scores, std::max_element(scores, scores + NUM_CLASSES));
        if (pred == test_data.labels[start + i]) final_correct++;
    }
}

double final_acc = 100.0 * final_correct / n_test;
std::cout << "\n===========================================\n";
std::cout << "     FINAL TEST ACCURACY = " << final_acc << "%\n";
std::cout << "===========================================\n\n";


    // Free
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFree(d_Y);
    cudaFree(d_update_count);

    return 0;
}
