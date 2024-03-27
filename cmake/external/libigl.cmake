if (TARGET igl::core)
    return()
endif ()

include(FetchContent)
FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(libigl)

if (NOT TARGET igl::core)
    message(FATAL_ERROR "Creation of target 'igl::core' failed")
endif ()

if(ENABLE_IMGUI_VIEWER)
    igl_include(glfw)
    if(NOT TARGET igl::glfw)
        message(FATAL_ERROR "Creation of target 'igl::glfw' failed")
    endif ()

    igl_include(imgui)
    if(NOT TARGET igl::imgui)
        message(FATAL_ERROR "Creation of target 'igl::imgui' failed")
    endif ()
endif ()