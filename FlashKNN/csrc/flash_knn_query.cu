#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <assert.h>


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <thrust/extrema.h>

#include "flash_knn_query.h"

int get_cuda_shared_mem(){
    return at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
}
