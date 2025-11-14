#include "cifar10_loader.h"
#include <fstream>
#include <iostream>

CIFAR10Data loadCIFAR10Batch(const std::string &filename, int num_images)
{
    CIFAR10Data data;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: no se pudo abrir" << filename << std::endl;
        return data;
    }

    const int image_size = 3072;
    for (int i = 0; i < num_images; i++)
    {
        unsigned char label;
        file.read((char *)&label, 1);

        std::vector<unsigned char> buffer(image_size);
        file.read((char *)buffer.data(), image_size);

        std::vector<float> image(image_size);
        for (int j = 0; j < image_size; j++)
        {
            image[j] = buffer[j] / 255.0f;
        }

        data.images.push_back(image);
        data.labels.push_back(label);
    }
    return data;
}