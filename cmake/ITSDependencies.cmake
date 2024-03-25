if (NOT TARGET Eigen3::Eigen)
    include(eigen)
endif ()

if (NOT TARGET igl::core)
    include(libigl)
endif ()

if (NOT TARGET CLI11::CLI11)
    include(CLI)
    target_compile_definitions(CLI11 INTERFACE -DCLI11_STD_OPTIONAL=0)
    target_compile_definitions(CLI11 INTERFACE -DCLI11_EXPERIMENTAL_OPTIONAL=0)
endif ()

if (NOT TARGET spdlog::spdlog)
    include(spdlog)
endif ()

if (FCPW_TEST)
    if (NOT TARGET fcpw)
        include(fcpw)
#        add_subdirectory(${CMAKE_SOURCE_DIR}/fcpw)
#        add_library(fcpw::fcpw ALIAS fcpw)
    endif ()
endif ()