###########################################
# Global file
#add_lib(ITS_CONFIG INTERFACE)
#
#target_sources(ITS_CONFIG
#        PRIVATE
#        Config.hpp
#)
#
#target_include_directories(ITS_CONFIG
#        INTERFACE
#        ${CMAKE_CURRENT_SOURCE_DIR}
#)
#target_link_libraries(ITS_CONFIG
#        INTERFACE
#        Eigen3::Eigen
#        spdlog::spdlog
#)

###########################################
# Detail folder
add_subdirectory(detail)
#link_libraries(ITS::DETAIL)

###########################################
# Marching Cubes
add_subdirectory(mc)

###########################################
# Core folder
add_subdirectory(core)

###########################################
# Organize linkings
target_link_libraries(ITS_GPU_MC
        PRIVATE
        ITS::SVO
        # ITS::MODEL
)

target_link_libraries(ITS_MODEL
        PRIVATE
        ITS::GPU_MC
)
