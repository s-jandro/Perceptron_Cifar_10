#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "cifar10_loader.h"

#define THREADS_PER_BLOCK 256
#define BATCH_SIZE 256

__global__ void forwardBatchKernel(const float *X, const float *w, float b, float *z,
                                   int batch_size, int input_size)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < batch_size) {

        // FORWARD: z = w^T * x + b
        float sum = b;
        const float *x = X + sample * input_size;
        for (int k = 0; k < input_size; ++k) {
            sum += x[k] * w[k];
        }
        z[sample] = sum;
    }
}

__device__ int step_function_gpu(float val, float threshold)
{
    // STEP FUNCTION: y_pred = (z > threshold)
    return val > threshold ? 1 : 0;
}

__global__ void rosenblattUpdateKernel(float *w, float *b, const float *X, const int *Y_labels,
                                       const float *z, int batch_size, int input_size,
                                       float learning_rate, float threshold, int *d_update_count)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch_size) return;

    int y_pred = step_function_gpu(z[sample], threshold);
    int y_true = Y_labels[sample];
    if (y_pred == y_true) return;

    int error = y_true - y_pred;
    float lr_error = learning_rate * (float)error;
    const float *x = X + sample * input_size;

    // ROSENBLATT UPDATE: w += lr * error * x
    for (int k = 0; k < input_size; ++k) {
        atomicAdd(&w[k], lr_error * x[k]);
    }

    // ROSENBLATT UPDATE: b += lr * error
    atomicAdd(b, lr_error);

    atomicAdd(d_update_count, 1);
}

void save_checkpoint(const std::string &file, const std::vector<float> &w, float b)
{
    std::ofstream ofs(file, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&b), sizeof(float));
    ofs.close();
}

int main()
{
    std::cout << "Perceptrón Rosenblatt CLÁSICO (Binario) - CUDA\n";
    std::cout << "Arquitectura: Entrada (3072) → Pesos (3072) → Umbral → Salida (1)\n\n";

    const int INPUT_SIZE = 3072;
    const int EPOCHS = 100;
    float LEARNING_RATE = 0.01f;
    const float THRESHOLD = 0.0f;
    const int CHECKPOINT_INTERVAL = 50;

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

    std::cout << "Train samples: " << n_train << " Test samples: " << n_test << '\n';

    std::vector<int> train_labels_binary(n_train);
    std::vector<int> test_labels_binary(n_test);

    for (int i = 0; i < n_train; ++i) {
        train_labels_binary[i] = (train_data.labels[i] < 4) ? 0 : 1;
    }
    for (int i = 0; i < n_test; ++i) {
        test_labels_binary[i] = (test_data.labels[i] < 4) ? 0 : 1;
    }

    std::cout << "Problema binario: [0,1,2,3] → 0 | [4,5,6,7,8,9] → 1\n";
    std::cout << "EPOCHS: " << EPOCHS << " LR: " << LEARNING_RATE << " THRESHOLD: " << THRESHOLD << "\n\n";

    std::vector<float> w(INPUT_SIZE);
    float b = 0.0f;

    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto &weight : w) weight = dist(gen);

    float *d_w = nullptr, *d_b_ptr = nullptr;
    float *d_X = nullptr, *d_z = nullptr;
    int *d_Y = nullptr, *d_update_count = nullptr;

    cudaMalloc(&d_w, w.size() * sizeof(float));
    cudaMalloc(&d_b_ptr, sizeof(float));
    cudaMalloc(&d_X, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_z, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_Y, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_update_count, sizeof(int));

    cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_ptr, &b, sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> X_batch(BATCH_SIZE * INPUT_SIZE);
    std::vector<int> Y_batch(BATCH_SIZE);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto ep_start = std::chrono::high_resolution_clock::now();
        int total_updates = 0;

        std::vector<int> idx(n_train);
        for (int i = 0; i < n_train; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), gen);

        int num_batches = (n_train + BATCH_SIZE - 1) / BATCH_SIZE;
        for (int bidx = 0; bidx < num_batches; ++bidx) {
            int start = bidx * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, n_train);
            int current_bs = end - start;

            for (int i = 0; i < current_bs; ++i) {
                const auto &img = train_data.images[idx[start + i]];
                std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
                Y_batch[i] = train_labels_binary[idx[start + i]];
            }

            if (current_bs < BATCH_SIZE) {
                std::fill(X_batch.begin() + current_bs * INPUT_SIZE, X_batch.end(), 0.0f);
                std::fill(Y_batch.begin() + current_bs, Y_batch.end(), 0);
            }

            cudaMemcpy(d_X, X_batch.data(),
                       current_bs * INPUT_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(d_Y, Y_batch.data(),
                       current_bs * sizeof(int),
                       cudaMemcpyHostToDevice);

            int blocks_forward = (current_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            // FORWARD (TRAIN): z = w^T * x + b
            forwardBatchKernel<<<blocks_forward, THREADS_PER_BLOCK>>>(
                d_X, d_w, b, d_z, current_bs, INPUT_SIZE
            );

            cudaMemset(d_update_count, 0, sizeof(int));

            int blocks_update = (current_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            // ROSENBLATT UPDATE
            rosenblattUpdateKernel<<<blocks_update, THREADS_PER_BLOCK>>>(
                d_w, d_b_ptr, d_X, d_Y, d_z, current_bs,
                INPUT_SIZE, LEARNING_RATE, THRESHOLD, d_update_count
            );

            cudaDeviceSynchronize();

            int updates_batch = 0;
            cudaMemcpy(&updates_batch, d_update_count, sizeof(int), cudaMemcpyDeviceToHost);
            total_updates += updates_batch;
        }

        auto ep_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                ep_end - ep_start).count() / 1000.0;

        int correct = 0;
        std::vector<float> z_host(BATCH_SIZE);

        for (int tb = 0; tb < (n_test + BATCH_SIZE - 1) / BATCH_SIZE; ++tb) {
            int start = tb * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, n_test);
            int cur_bs = end - start;

            for (int i = 0; i < cur_bs; ++i) {
                const auto &img = test_data.images[start + i];
                std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
            }

            if (cur_bs < BATCH_SIZE) {
                std::fill(X_batch.begin() + cur_bs * INPUT_SIZE, X_batch.end(), 0.0f);
            }

            cudaMemcpy(d_X, X_batch.data(),
                       cur_bs * INPUT_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice);

            int blocks_eval = (cur_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            // FORWARD (TEST): z = w^T * x + b
            forwardBatchKernel<<<blocks_eval, THREADS_PER_BLOCK>>>(
                d_X, d_w, b, d_z, cur_bs, INPUT_SIZE
            );

            cudaDeviceSynchronize();

            cudaMemcpy(z_host.data(), d_z,
                       cur_bs * sizeof(float),
                       cudaMemcpyDeviceToHost);

            for (int i = 0; i < cur_bs; ++i) {
                int y_pred = z_host[i] > THRESHOLD ? 1 : 0;
                if (y_pred == test_labels_binary[start + i]) correct++;
            }
        }

        double acc = 100.0 * correct / n_test;
        std::cout << "Epoch " << (epoch + 1)
                  << "/" << EPOCHS
                  << " | updates: " << total_updates
                  << " | accuracy: " << acc << "%"
                  << " | time: " << epoch_time << "s\n";

        if ((epoch + 1) % CHECKPOINT_INTERVAL == 0 || epoch == EPOCHS - 1) {
            cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&b, d_b_ptr, sizeof(float), cudaMemcpyDeviceToHost);
            std::string fname = "rosenblatt_cuda_checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
            save_checkpoint(fname, w, b);
            std::cout << " → Checkpoint saved: " << fname << '\n';
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count();

    std::cout << "\n==========================================\n";
    std::cout << " Evaluando precisión FINAL...\n";
    std::cout << "==========================================\n";

    cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, d_b_ptr, sizeof(float), cudaMemcpyDeviceToHost);

    int final_correct = 0;
    std::vector<float> z_host_final(BATCH_SIZE);

    for (int tb = 0; tb < (n_test + BATCH_SIZE - 1) / BATCH_SIZE; ++tb) {
        int start = tb * BATCH_SIZE;
        int end = std::min(start + BATCH_SIZE, n_test);
        int cur_bs = end - start;

        for (int i = 0; i < cur_bs; ++i) {
            const auto &img = test_data.images[start + i];
            std::copy(img.begin(), img.end(), X_batch.begin() + i * INPUT_SIZE);
        }

        if (cur_bs < BATCH_SIZE) {
            std::fill(X_batch.begin() + cur_bs * INPUT_SIZE, X_batch.end(), 0.0f);
        }

        cudaMemcpy(d_X, X_batch.data(),
                   cur_bs * INPUT_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        int blocks_final = (cur_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // FORWARD (FINAL): z = w^T * x + b
        forwardBatchKernel<<<blocks_final, THREADS_PER_BLOCK>>>(
            d_X, d_w, b, d_z, cur_bs, INPUT_SIZE
        );

        cudaDeviceSynchronize();

        cudaMemcpy(z_host_final.data(), d_z,
                   cur_bs * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < cur_bs; ++i) {
            int y_pred = z_host_final[i] > THRESHOLD ? 1 : 0;
            if (y_pred == test_labels_binary[start + i]) final_correct++;
        }
    }

    double final_acc = 100.0 * final_correct / n_test;

    std::cout << "\nFINAL TEST ACCURACY: " << final_acc << "%\n";
    std::cout << "Correct: " << final_correct << " / " << n_test << "\n";
    std::cout << "Total time: " << total_time << " seconds\n";
    std::cout << "==========================================\n\n";

    save_checkpoint("rosenblatt_cuda_final.bin", w, b);

    cudaFree(d_w);
    cudaFree(d_b_ptr);
    cudaFree(d_X);
    cudaFree(d_z);
    cudaFree(d_Y);
    cudaFree(d_update_count);

    return 0;
}
