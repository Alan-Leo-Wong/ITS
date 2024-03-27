add_lib(ITS_VIEWER)

target_sources(ITS_VIEWER
        PRIVATE
        MeshViewer.hpp
        MeshViewer.cpp
)

target_link_libraries(ITS_VIEWER
        PUBLIC
        igl::core
        igl::glfw
        igl::imgui
)
