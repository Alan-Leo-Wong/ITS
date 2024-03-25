###########################################
# common computation on GPU for other libs
add_cuda_lib(ITS_GPU_COMPUTE STATIC)
target_sources(ITS_GPU_COMPUTE
        PRIVATE
        CUDACompute.hpp
        CUDACompute.cu
)
target_link_libraries(ITS_GPU_COMPUTE
        PUBLIC
        CUDA::cublas
        Eigen3::Eigen
)

###########################################
# GPU SVO
add_cuda_lib(ITS_SVO STATIC)
target_sources(ITS_SVO
        PRIVATE
        MortonLUT.hpp
        SVO.hpp
        SVO.cu
)
target_link_libraries(ITS_SVO
        PUBLIC
        ITS::GPU_COMPUTE
)

###########################################
# ITS model
add_lib(ITS_MODEL STATIC)
target_sources(ITS_MODEL
        PRIVATE
        BSpline.hpp
        Mesh.hpp
        Mesh.cpp
        ThinShells.hpp
        ThinShells.cpp
)
target_link_libraries(ITS_MODEL
        PUBLIC
        ITS::SVO
        igl::core
)
target_compile_options(ITS_MODEL
        PRIVATE
        ${OpenMP_CXX_FLAGS}
)
if (MSVC)
    target_compile_options(ITS_MODEL
            PRIVATE
            "-openmp:experimental"
    )
endif ()