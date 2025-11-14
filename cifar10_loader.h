#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <vector>
#include <string>

struct CIFAR10Data
{
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

CIFAR10Data loadCIFAR10Batch(const std::string &filename, int num_images = 10000);

#endif
