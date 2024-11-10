add_cuda_lib(ITS_GPU_MC STATIC)

target_sources(ITS_GPU_MC
        PRIVATE
        MarchingCubesLUT.hpp
        MarchingCubes.hpp
        MarchingCubes.cu
)

target_compile_options(ITS_GPU_MC PRIVATE -diag-suppress=177 -diag-suppress=20012 -diag-suppress=20015 -diag-suppress=20013)
if(MSVC)
    target_compile_options(ITS_GPU_MC PRIVATE -Xcompiler /wd4068)
endif ()

#target_link_libraries(ITS_GPU_MC
#        PRIVATE
#        spdlog::spdlog
#)

#target_compile_options(ITS_GPU_MC
#        PRIVATE
#        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++17>
#)
