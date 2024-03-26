add_executable(ITS_MAIN main.cpp)

target_link_libraries(ITS_MAIN
        PRIVATE
        ITS::MODEL
        CLI11::CLI11
)

if (FCPW_TEST)
    target_sources(ITS_MAIN
            PRIVATE
            FCPWHelper.hpp
    )
    target_link_libraries(ITS_MAIN
            PRIVATE
            fcpw::fcpw
            Eigen3::Eigen
    )
    target_include_directories(ITS_MAIN
            PRIVATE
#            ${FCPW_EIGEN_INCLUDES}
            ${FCPW_ENOKI_INCLUDES}
    )
    target_compile_definitions(ITS_MAIN PRIVATE -DFCPW_TEST)
endif ()