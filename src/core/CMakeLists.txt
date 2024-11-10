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
        #        ITS::CONFIG
#        spdlog::spdlog
        Eigen3::Eigen
        CUDA::cublas
)
target_compile_options(ITS_GPU_COMPUTE PRIVATE -diag-suppress=177 -diag-suppress=20012 -diag-suppress=20015 -diag-suppress=20013)
if(MSVC)
    target_compile_options(ITS_GPU_COMPUTE PRIVATE -Xcompiler /wd4068)
endif ()

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
#        spdlog::spdlog
)

###########################################
# ITS model
add_lib(ITS_BASE_MODEL STATIC)
target_sources(ITS_BASE_MODEL
        PRIVATE
        Mesh.hpp
        Mesh.cpp
)
target_compile_options(ITS_BASE_MODEL PUBLIC /wd4068 /wd4384)
target_link_libraries(ITS_BASE_MODEL
        PUBLIC
        igl::core
        ITS::GPU_COMPUTE
        spdlog::spdlog
)

if (MSVC)
    target_compile_options(ITS_BASE_MODEL
            PUBLIC
            "-openmp:experimental"
    )
else ()
    target_compile_options(ITS_BASE_MODEL
            PUBLIC
            ${OpenMP_CXX_FLAGS}
    )
endif ()

add_lib(ITS_MODEL STATIC)
target_sources(ITS_MODEL
        PRIVATE
        BSpline.hpp
        ThinShells.hpp
        ThinShells.cpp
)
target_link_libraries(ITS_MODEL
        PUBLIC
        ITS::SVO
        ITS::BASE_MODEL
#        spdlog::spdlog
)