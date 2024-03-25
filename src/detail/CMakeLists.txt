#add_lib(ITS_DETAIL INTERFACE)
#
#target_sources(ITS_DETAIL
#        PRIVATE
#        BasicDataType.h
#)
#
##target_include_directories(ITS_DETAIL
##        INTERFACE
##        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
##        $<INSTALL_INTERFACE:include>)
#
#target_link_libraries(ITS_DETAIL
#        INTERFACE
#        Eigen3::Eigen
#)