###########################################
# Root CMakeList
###########################################

# General information
cmake_minimum_required(VERSION 3.20)
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

###########################################
# Options
option(FCPW_TEST "FCPW test" ON)

###########################################
# Project settings
project(ITS LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

# RelWithDebug for MSVC
if (MSVC)
    # https://github.com/mozilla/sccache/issues/242
    if (CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "sccache")
        string(REGEX REPLACE "/Z[iI7]" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")
    endif ()
endif ()


###########################################
# CMake module path
list(PREPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake/
        ${CMAKE_CURRENT_LIST_DIR}/cmake/common
        ${CMAKE_CURRENT_LIST_DIR}/cmake/external)

# Include helper functions
include(add_lib)

###########################################
# 3rdParty
include(ITSDependencies)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# CUDAToolkit
find_package(CUDAToolkit REQUIRED)

###########################################
# Sub directories
include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)
add_subdirectory(app)
