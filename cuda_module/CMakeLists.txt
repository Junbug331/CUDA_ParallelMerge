cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_STANDARD 14)
project(cuda_merge CUDA)

set(CUDA_MERGE_SRC
    src/cuda_merge_kernel.cu
    src/cuda_merge.cu
)

add_library(cuda_merge ${CUDA_MERGE_SRC})

target_include_directories(cuda_merge PUBLIC
    include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(cuda_merge PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
