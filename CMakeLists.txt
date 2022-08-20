cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
enable_language(CUDA)
project(CUDA_ParallelMerge VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

find_package(spdlog REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/install/lib/cmake/spdlog)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/spdlog/install/include)

add_subdirectory(cuda_module)

add_executable(${TARGET} src/main.cpp)
target_link_libraries(${TARGET} PRIVATE
        cuda_merge
        spdlog::spdlog)
set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_custom_target(copy-compile-commands ALL
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_LIST_DIR}
)
