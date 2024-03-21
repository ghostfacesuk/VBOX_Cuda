#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>

// Define the struct for storing data
struct DataPoint {
    float sats;
    float time;
    float lat;
    float lon;
    float velocity;
    float heading;
    float height;
    float vertVel;
    float tsample;
    int solutionType;
    int avifileindex;
    int avitime;
    float pps_mim;
    float teensyCount;
};

// CUDA kernel to process data
__global__ void processData(DataPoint* data, int numPoints, int totalBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = totalBlocks * blockDim.x;

    for (int i = idx; i < numPoints; i += stride) {
        // Process data[i]
        printf("Thread ID: %d, Time: %.3f, Lat: %.8f, Lon: %.8f, Velocity: %.3f\n",
            i, data[i].time, data[i].lat, data[i].lon, data[i].velocity);
    }
}

int main() {
    // Read data from the .vbo file
    std::ifstream file("data.vbo");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    // Skip the column names
    std::string line;
    std::getline(file, line);

    // Read data points
    std::vector<DataPoint> dataPoints;
    DataPoint point;
    while (file >> point.sats >> point.time >> point.lat >> point.lon >> point.velocity >>
        point.heading >> point.height >> point.vertVel >> point.tsample >> point.solutionType >>
        point.avifileindex >> point.avitime >> point.pps_mim >> point.teensyCount) {
        dataPoints.push_back(point);
    }
    file.close();

    // Print debug information about the data points
    std::cout << "Number of data points: " << dataPoints.size() << std::endl;
    for (int i = 0; i < std::min(static_cast<int>(dataPoints.size()), 10); ++i) {
        std::cout << "Data point " << i << ": " << dataPoints[i].time << ", " << dataPoints[i].lat << ", " << dataPoints[i].lon << ", " << dataPoints[i].velocity << std::endl;
    }

    // Prepare data for GPU
    DataPoint* d_data;
    cudaMalloc(&d_data, dataPoints.size() * sizeof(DataPoint));
    cudaMemcpy(d_data, dataPoints.data(), dataPoints.size() * sizeof(DataPoint), cudaMemcpyHostToDevice);

    // Define CUDA kernel configuration
    int blockSize = 128;
    const int MAX_BLOCKS = 65536;
    int numBlocks = (dataPoints.size() + blockSize - 1) / blockSize;
    if (numBlocks > MAX_BLOCKS) {
        numBlocks = MAX_BLOCKS;
    }
    int totalBlocks = (dataPoints.size() + blockSize * numBlocks - 1) / (blockSize * numBlocks);

    // Launch CUDA kernel to process data
    for (int i = 0; i < totalBlocks; ++i) {
        processData << <numBlocks, blockSize >> > (d_data, dataPoints.size(), totalBlocks);
        cudaDeviceSynchronize();

        // Check for kernel launch errors
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;
            return 1;
        }

        // Check for kernel execution errors
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(cudaError) << std::endl;
            return 1;
        }
    }

    // Flush any buffered output from the kernel
    cudaDeviceReset();

    // Clean up
    cudaFree(d_data);

    return 0;
}