# ITS: Implicit 3D Thin Shell

ITS introduces a novel approach to representing the "sandwich-walled" space of input surfaces using a tri-variate tensor-product B-spline. This method transforms complex geometric calculations into manageable operations, enhancing both the accuracy and efficiency of geometric processing tasks.

## Key Features

- **Novel Representation**: Utilizes a tri-variate tensor-product B-spline to express the implicit function of input surfaces, enabling precise manipulations and analyses.
- **Optimized Extreme Value Calculation**: Converts the challenge of identifying extreme function values across infinite points into a tractable problem of finding extremes within a finite set of candidates, ensuring rigorous wrapping of the input surface.
- **Acceleration Strategies**: Implements several optimization strategies that significantly reduce computation times for inside-outside tests, enhancing the overall efficiency.
- **Advanced Visualization Tools**: Features robust visualization capabilities implemented using [libigl](https://github.com/libigl/libigl), allowing for real-time viewing and analysis of geometric data.

## Prerequisites

Before installing and running ITS, ensure your system meets the following requirements:

- **NVIDIA GPU**: An NVIDIA GPU with CUDA Compute Capability 6.0 or higher.
- **CUDA Toolkit**: NVIDIA CUDA Toolkit version 11.6 or higher. Make sure it is properly installed and configured in your system environment.
- **C++ Compiler**: A compatible C++ compiler that supports at least C++20. This is necessary for compiling the project.
- **CMake**: Version 3.20 or higher for building the project.

## Installation

ITS requires CUDA and CMake for compilation. Follow these steps to set up and build the project:

1. **Clone the Repository**
   
   ```bash
   git clone https://github.com/Alan-Leo-Wong/ITS.git
   cd ITS
   ```
   
1. **Build the Project Using CMake**
   
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build . -j your-core-num

## Usage

To run ITS, use the following command-line interface (CLI) to configure your processing pipeline:

```
./ITS_MAIN -f <input_mesh_file> [options]
```

### CLI Options

- `-f, --in_file <file>`: Specifies the input mesh file **(required)**.
- `-U, --mesh_norm`: Normalize the input mesh.
- `-N, --mesh_noise`: Add noise to the mesh.
- `-P, --mesh_noise_per <percentage>`: Specifies the percentage of noise to add to the mesh. Requires `-N` flag and the percentage must be greater than zero.
- `-r, --svo_res <resolution>`: Specifies the resolution of the sparse voxel octree **(required)**.
- `-M, --mc`: Enable marching-cubes visualization.
- `-R, --mc_res <resolution>`: Specifies the resolution for marching-cubes. Requires `-M` flag and the resolution must be greater than zero.
- `-O, --mc_dir <directory>`: Specifies the output directory for marching-cubes visualization shells. Requires `-M` flag.
