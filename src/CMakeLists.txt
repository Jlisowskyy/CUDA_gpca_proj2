cmake_minimum_required(VERSION 3.29)
project(HammingOne CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# ------------------------------
# Cuda sources config
# ------------------------------

file(GLOB CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(CUDA_LIB OBJECT ${CUDA_SOURCES})

# CUDA Architecture setting
set_property(TARGET CUDA_LIB PROPERTY CUDA_ARCHITECTURES 60)

target_include_directories(CUDA_LIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib")
target_include_directories(CUDA_LIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")

# ------------------------------
# CXX sources config
# ------------------------------

file(GLOB CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

add_executable(hamming ${CXX_SOURCES})

target_include_directories(hamming PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpu")
target_include_directories(hamming PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")

target_link_libraries(hamming PRIVATE CUDA_LIB)

# ------------------------------
# Flags config
# ------------------------------

# Compilation options for different build types
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "MSVC is unsupported...")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "CLANG is unsupported...")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("Using GCC...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
        )

        set_source_files_properties(
                ${CXX_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-O3;-march=native"
        )

    else ()
        message(FATAL_ERROR "Unsupported compiler.")
    endif ()
else ()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "MSVC is unsupported...")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "CLANG is unsupported...")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("Using GCC...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-g3;-fno-omit-frame-pointer;-g;-G;-O0"
        )

        set_source_files_properties(
                ${CXX_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-O0;-march=native;-g;"
        )

    else ()
        message(FATAL_ERROR "Unsupported compiler.")
    endif ()
endif ()

