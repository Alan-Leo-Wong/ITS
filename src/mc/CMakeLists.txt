add_cuda_lib(ITS_GPU_MC STATIC)

target_sources(ITS_GPU_MC
        PRIVATE
        LookTable.hpp
        MarchingCubes.hpp
        MarchingCubes.cu
)
