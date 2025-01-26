import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
# 添加上一层目录到模块搜索路径
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

def gather(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    outTensor = inputTensor[tuple(indices)]
    return outTensor
    # return torch.gather(inputTensor, dim=axis, index=indexTensor)


def test(inputShape, indexShape, axis, test_dtype, device):
    print(
        f"Testing Softmax on {device} with x_shape:{inputShape} , indice_shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}"
    )
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index = np.random.randint(0, inputShape[axis], indexShape).astype(np.int32)
    indexTensor = torch.from_numpy(index).to(torch.int64).to(device)
    
    # if axis == 0 and test_dtype == torch.float32:
        # print("Input tensor:\n", inputTensor)
        # print("\nIndex tensor:\n", indexTensor)

    rank = len(inputShape)
    outTensor = gather(rank, axis, inputTensor, indexTensor)

    Q_output = torch.zeros(outTensor.shape, device=device, dtype=test_dtype)
    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    output_ptr = ctypes.cast(Q_output.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    
    # input_J, input_K = inputShape[0], inputShape[1]
    # input_I = inputShape[2] if len(inputShape) > 2 else 1
    # index_J, index_K = indexShape[0], indexShape[1]
    # index_I = indexShape[2] if len(indexShape) > 2 else 1
    if len(inputShape) == 2:
        input_I, input_J = inputShape[0], inputShape[1]
    elif len(inputShape) == 3:
        input_I, input_J, input_K = inputShape[0], inputShape[1], inputShape[2]
    
    if len(indexShape) == 2:
        index_I, index_J = indexShape[0], indexShape[1]
    elif len(indexShape) == 3:
        index_I, index_J, index_K = indexShape[0], indexShape[1], indexShape[2]
    
    

    # print(inputTensor, indexTensor, outTensor)
    if 1:
        if test_dtype == torch.float32:
            # 将元组转换为 ctypes 数组
            inputShape_length = len(inputShape)
            indexShape_length = len(indexShape)
            
            inputShape = (ctypes.c_int * len(inputShape))(*inputShape)
            indexShape = (ctypes.c_int * len(indexShape))(*indexShape) 

            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))

            lib.gather_gpu_f32_Nd.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    
                ]
            
            custom_gather_time = performance.CudaProfile((
                lib.gather_gpu_f32_Nd,
                (input_ptr, index_ptr, output_ptr, axis, inputShape_length, indexShape_length, inputShape, indexShape)
            ))

        if test_dtype == torch.float16:
            # 将元组转换为 ctypes 数组
            inputShape_length = len(inputShape)
            indexShape_length = len(indexShape)
            
            inputShape = (ctypes.c_int * len(inputShape))(*inputShape)
            indexShape = (ctypes.c_int * len(indexShape))(*indexShape) 

            torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))

            lib.gather_gpu_f16_Nd.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    
                ]
            
            custom_gather_time = performance.CudaProfile((
                lib.gather_gpu_f16_Nd,
                (input_ptr, index_ptr, output_ptr, axis, inputShape_length, indexShape_length, inputShape, indexShape)
            ))
    # else:
    #     if test_dtype == torch.float32:
    #         if device == "cuda":
    #             torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))

    #             lib.gather_gpu_f32.argtypes = [
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int
    #             ]

    #             custom_gather_time = performance.CudaProfile((
    #                 lib.gather_gpu_f32,
    #                 (input_ptr, index_ptr, output_ptr,axis, input_I, input_J, index_I, index_J)
    #             ))
                
    #     if test_dtype == torch.float16:
    #         if device == "cuda":
    #             torch_gather_time = performance.CudaProfile((gather, (rank, axis, inputTensor, indexTensor)))
    #             lib.gather_gpu_f16.argtypes = [
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.POINTER(ctypes.c_void_p),
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int,
    #                 ctypes.c_int
    #             ]
    #             custom_gather_time = performance.CudaProfile((
    #                 lib.gather_gpu_f16,
    #                 (input_ptr, index_ptr, output_ptr, axis, input_I, input_J, index_I, index_J)
    #             ))
    performance.logBenchmark(torch_gather_time, custom_gather_time)

    tmpa = outTensor.to('cpu').numpy().flatten()
    print(tmpa) #torch算的答案
    tmpb = Q_output.to('cpu').numpy().flatten()
    print(tmpb)

    atol = max(abs(tmpa - tmpb))

    rtol = atol / max(abs(tmpb) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
parser = argparse.ArgumentParser(description="Test softmax on different devices.")
parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
args = parser.parse_args()    
test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((12000, 2000), (2, 2), 1, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),
        ((3, 2, 4, 10), (1, 2, 3, 4), 0, torch.float16, "cuda"),
        ((1, 2, 1, 10, 20), (5, 4, 3, 2, 1), 4, torch.float16, "cuda"),
        ((3, 2, 4, 10), (1, 2, 3, 4), 0, torch.float32, "cuda"),
        ((1, 2, 1, 10, 20), (5, 4, 3, 2, 1), 4, torch.float32, "cuda"),
         
]
filtered_test_cases = [
    (inputShape , indexShape, axis, test_dtype, device)
    for inputShape , indexShape, axis, test_dtype, device in test_cases
    if device == args.device
]
if args.device == 'mlu':
    import torch_mlu
# 执行过滤后的测试用例
for inputShape , indexShape, axis, test_dtype, device in filtered_test_cases:
    test(inputShape , indexShape, axis, test_dtype, device)