#include <cstdio>
#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <iostream>
#include <unistd.h>


template <typename T>
__global__ void gather_ND_Alligned(T *data, long long *indices, T *output, int input_I, int input_J, int input_K, int indicesSize) {
        
    constexpr auto offset = (std::is_same_v<T, float>) ? 4 : 2;
    // 4 = sizeof(float4) / sizeof(float)
    // 2 = sizeof(half2) / sizeof(half)

    // k代表vz方向的线程数 当offset=4时，一个线程负责4个float的访存
    // 即k=0负责data[0...3]的访存，依次类推
    int k = (threadIdx.z + blockIdx.z * blockDim.z) * offset;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < input_I && j < indicesSize && k < input_K) {
        const long long index = indices[j];

        // 注意到，i * input_J * input_K + index * input_K + k 显然若inpout_K为offset倍数时，原式一定可以被offset整除 即对齐内存
        if (k + offset <= input_K) {
            if constexpr (std::is_same_v<T, float>) (float4& )output[i * indicesSize * input_K + j * input_K + k] = (float4& )data[i * input_J * input_K + index * input_K + k];
            else (half2 &)output[i * indicesSize * input_K + j * input_K + k] = (half2& )data[i * input_J * input_K + index * input_K + k];
        } 

        else {
            for(int m = k; m < input_K; m++) {
                output[i * indicesSize * input_K + j * input_K + k] = data[i * input_J * input_K + index * input_K + k];
            }
        }
    }
}


template <typename T>
__global__ void gather_ND_unAlligned(T *data, long long *indices, T *output, int input_I, int input_J, int input_K, int indicesSize) {

    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 如果内存未对齐 则使用最朴素的访存方式
    if (i < input_I && j < indicesSize && k < input_K) {
        long long index = indices[j];
        output[i * indicesSize * input_K + j * input_K + k] = data[i * input_J * input_K + index * input_K + k];
    }
}



template <typename T, uint axis>
__global__ void gather_2D(T * const input, long long * const indices, T *output, const int input_I, const int input_J, const int index_I, const int index_J) {
    if constexpr (axis == 1) {
        const uint z = threadIdx.z + blockIdx.z * blockDim.z;
        const uint y = threadIdx.y + blockIdx.y * blockDim.y;
        const uint x = threadIdx.x + blockIdx.x * blockDim.x;
        //output[x, y, z] = input[x, indices[y, z]]

        // v1
        // 这种访存方式保留了在vz方向的连续访存，更加高效
        if (x < input_I && y < index_I && z < index_J) {
            const int index = indices[y * index_J + z];
            output[x * index_J * index_I + y * index_J + z] = input[x * input_J + index];
        }
    }

    else if constexpr (axis == 0) {
        constexpr auto offset = (std::is_same_v<T, float>) ? 4 : 2;

        int z = (threadIdx.z + blockIdx.z * blockDim.z) * offset;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int x = (threadIdx.x + blockIdx.x * blockDim.x);

        // output[x, y, z] = data[[indices[x, y], z]
        if (x < index_I && y < index_J && z < input_J){
            const int index = indices[x * index_J + y];

            if (z + offset <= input_J) {
                if constexpr (std::is_same_v<T, float>) {(float4 &)output[x*(index_J*input_J) + y*input_J + z] = (float4 &)input[index*input_J + z];}
                else {(half2 &)output[x*(index_J*input_J) + y*input_J + z] = (half2 &)input[index*input_J + z];}
            } 
            else {
                for(int i = z; i < input_J; i++) {
                    output[x*(index_J*input_J) + y*input_J + i] = input[index*input_J + i];
                }
            }
        }
    }
}


// gather_gpu_f32和gather_gpu_f16只能处理两维的情况
extern "C" void gather_gpu_f32(void const *data, void const *indices, void *output, int axis, int input_I, int input_J, int index_I, int index_J)
{
    if (axis == 1) {
        dim3 blockSize(1, 2, 64);
        dim3 gridSize(
            (input_I + blockSize.x - 1) / blockSize.x,
            (index_I + blockSize.y - 1) / blockSize.y,
            (index_J + blockSize.z - 1) / blockSize.z
        );
        gather_2D<float, 1><<<gridSize, blockSize>>>((float *)data, (long long *)indices, (float *)output, input_I, input_J, index_I, index_J);
        // 启动kernel的时候 一定要和torch端代码的数据类型指定为相同类型
        // cudaDeviceSynchronize();
    }

    else if (axis == 0) {
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
            (index_I + blockSize.x - 1) / blockSize.x,
            (index_J + blockSize.y - 1) / blockSize.y,
            (input_J + blockSize.z * 4 - 1) / (blockSize.z * 4)
        );
        gather_2D<float, 0><<<gridSize, blockSize>>>((float *)data, (long long *)indices, (float *)output, input_I, input_J, index_I, index_J);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
}



// gather_gpu_f32和gather_gpu_f16只能处理两维的情况
extern "C" void gather_gpu_f16(void const *data, void const *indices, void const *output, int axis, int input_I, int input_J, int index_I, int index_J)
{
    if (axis == 1) {
        dim3 blockSize(1, 4, 64);
        dim3 gridSize(
            (input_I + blockSize.x - 1) / blockSize.x,
            (index_I + blockSize.y - 1) / blockSize.y,
            (index_J + blockSize.z - 1) / blockSize.z
        );
        gather_2D<half, 1><<<gridSize, blockSize>>>((half *)data, (long long *)indices, (half *)output, input_I, input_J, index_I, index_J);
        // 启动kernel的时候 一定要和torch端代码的数据类型指定为相同类型
        // cudaDeviceSynchronize();
    }

    else if (axis == 0) {    
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
            (index_I + blockSize.x - 1) / blockSize.x,
            (index_J + blockSize.y - 1) / blockSize.y,
            (input_J + blockSize.z * 2 - 1) / (blockSize.z * 2)
        );
        gather_2D<half,0><<<gridSize, blockSize>>>((half *)data, (long long *)indices, (half *)output, input_I, input_J, index_I, index_J);

    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
}



// gather_gpu_f32_Nd和gather_gpu_f16_Nd支持任意维的gather操作
extern "C" void gather_gpu_f32_Nd(void const *data, void const *indices, void *output, int axis, int inputShape_length, int indexShape_length, int const* inputShape, int const* indexShape)
{
    //声明所需要的维度尺寸
    int indexSize = 1;
    int input_I = 1, input_J = 1, input_K = 1;

    //indexSize即indices矩阵的总体大小 即所有维度上的大小相乘
    for (auto i = 0; i < indexShape_length; ++i) indexSize *= indexShape[i];

    //将input矩阵的n维分为三部分，轴(axis)前，中和后
    //若axis=1或0，则退化为input_I=1或input_K=1的二维情况，此时也可以调用gather_gpu_f32处理
    for (auto i = 0; i < axis; ++i) input_I *= inputShape[i];
    input_J = inputShape[axis];
    for (auto i = axis + 1; i < inputShape_length; ++i) input_K *= inputShape[i];

    const bool isAligned = (input_K % 4 == 0); // 4 = sizeof(float4) / sizeof(float)

    //若未对齐 则使用未对齐的核函数
    //其中blockSize*gridSize应与input，indices两矩阵总大小乘积相等
    if (!isAligned) {
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
        (input_I + blockSize.x - 1) / blockSize.x,
        (indexSize + blockSize.y - 1) / blockSize.y,
        (input_K + blockSize.z - 1) / blockSize.z
        );
        gather_ND_unAlligned<float><<<gridSize, blockSize>>>((float*)data, (long long*)indices, (float*)output, input_I, input_J, input_K, indexSize);
    }

    //若对齐 则使用float4加速访存
    //由于这下一个线程可以读取并写入四个float，所以线程数量也可以减少到25%
    else if (isAligned) {
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
        (input_I + blockSize.x - 1) / blockSize.x,
        (indexSize + blockSize.y - 1) / blockSize.y,
        (input_K + blockSize.z * 4 - 1) / (blockSize.z * 4)
        );
        gather_ND_Alligned<float><<<gridSize, blockSize>>>((float*)data, (long long*)indices, (float*)output, input_I, input_J, input_K, indexSize);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
}



// 注释见gather_gpu_f32_Nd
extern "C" void gather_gpu_f16_Nd(void const *data, void const *indices, void *output, int axis, int inputShape_length, int indexShape_length, int const* inputShape, int const* indexShape)
{
    int indexSize = 1;
    int input_I = 1, input_J = 1, input_K = 1;

    for (auto i = 0; i < indexShape_length; ++i) indexSize *= indexShape[i];

    for (auto i = 0; i < axis; ++i) input_I *= inputShape[i];
    input_J = inputShape[axis];
    for (auto i = axis + 1; i < inputShape_length; ++i) input_K *= inputShape[i];

    const bool isAligned = (input_K % 2 == 0); // 2 = sizeof(half2) / sizeof(half)

    if (!isAligned) {
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
        (input_I + blockSize.x - 1) / blockSize.x,
        (indexSize + blockSize.y - 1) / blockSize.y,
        (input_K + blockSize.z - 1) / blockSize.z
        );
        gather_ND_unAlligned<half><<<gridSize, blockSize>>>((half*)data, (long long*)indices, (half*)output, input_I, input_J, input_K, indexSize);
    }

        
    else if (isAligned) {
        dim3 blockSize(1, 2, 32);
        dim3 gridSize(
        (input_I + blockSize.x - 1) / blockSize.x,
        (indexSize + blockSize.y - 1) / blockSize.y,
        (input_K + blockSize.z * 2 - 1) / (blockSize.z * 2)
        );
        gather_ND_Alligned<half><<<gridSize, blockSize>>>((half*)data, (long long*)indices, (half*)output, input_I, input_J, input_K, indexSize);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }
}