message(STATUS "Third-party: creating target 'fcpw::fcpw'")

include(FetchContent)
FetchContent_Declare(
        fcpw
        GIT_REPOSITORY https://github.com/rohan-sawhney/fcpw.git
        GIT_TAG master
)
FetchContent_GetProperties(fcpw)
if (NOT fcpw_POPULATED)
    FetchContent_Populate(fcpw)
    add_subdirectory(${fcpw_SOURCE_DIR} ${fcpw_BINARY_DIR})
endif ()

if (NOT TARGET fcpw)
    message(FATAL_ERROR "Creation of target 'fcpw::fcpw' failed")
else ()
    add_library(fcpw::fcpw ALIAS fcpw)
    target_link_libraries(fcpw
            INTERFACE
            Eigen3::Eigen
    )
    target_include_directories(fcpw
            INTERFACE
            ${FCPW_ENOKI_INCLUDES}
    )
endif ()