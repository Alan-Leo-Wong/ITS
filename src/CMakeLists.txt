###########################################
# detail folder
add_subdirectory(detail)
#link_libraries(ITS::DETAIL)

###########################################
# marching cubes
add_subdirectory(mc)

###########################################
# core folder
add_subdirectory(core)

###########################################
# organize linkers
target_link_libraries(ITS_GPU_MC
        PRIVATE
        ITS::SVO
        # ITS::MODEL
)

target_link_libraries(ITS_MODEL
        PRIVATE
        ITS::GPU_MC
)
