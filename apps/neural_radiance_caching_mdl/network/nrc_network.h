/*
    largely sourced from: https://github.com/shocker-0x15/GfxExp/blob/master/neural_radiance_caching/network_interface.h
    credit to GfxExp for their NRC implementation
*/

#pragma once

#include <cuda.h>
#include <tiny-cuda-nn/common.h>
#include <cuda_runtime.h>
#include <tiny-cuda-nn/config.h>
#include <memory>

enum class PositionEncoding {
    TriangleWave,
    HashGrid,
    Frequency
};

// EN: Isolate the tiny-cuda-nn into the cpp side by pimpl idiom to avoid the situation where
//     the entire sample program needs to be compiled via nvcc.
class NeuralRadianceCache {
    class Priv;
    Priv* m = nullptr;

public:
    NeuralRadianceCache();
    ~NeuralRadianceCache();

    void initialize(PositionEncoding posEnc, uint32_t numHiddenLayers, float learningRate);
    void finalize();

    void infer(CUstream stream, float* inputData, uint32_t numData, float* predictionData);
    void train(CUstream stream, float* inputData, float* targetData, uint32_t numData,
        float* lossOnCPU = nullptr);
};