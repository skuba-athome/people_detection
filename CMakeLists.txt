cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
#find_package(VTK REQUIRED)
#include_directories(${VTK_INCLUDE_DIRS})

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()
find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
rosbuild_add_executable(people_detection src/people_detection.cpp)
target_link_libraries(people_detection libvtkCommon.so libvtkFiltering.so libvtkRendering.so)
rosbuild_add_boost_directories()
rosbuild_link_boost(people_detection filesystem system)

#rosbuild_add_executable(people_detection_dear src/people_detection_dear.cpp)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(people_detection_dear filesystem system)



#rosbuild_add_executable(multi_people_detection src/multi_people_detection.cpp)
#target_link_libraries(multi_people_detection libvtkCommon.so libvtkFiltering.so libvtkRendering.so)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(multi_people_detection filesystem system)

#rosbuild_add_executable(test_people src/test_people.cpp)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(test_people filesystem system)

#link_directories(${VTK_LIBRARY_DIRS})
#target_link_libraries(test_people ${VTK_LIBRARIES})


#add_executable (ground_based_rgbd_people_detector src/main_ground_based_people_detection.cpp)
#rosbuild_add_executable(test_people src/test_people.cpp)
#target_link_libraries (test_people libvtkCommon.so)

#rosbuild_add_executable(people_detection src/people_detection.cpp)
#rosbuild_add_boost_directories()

#target_link_libraries(test_people libvtkCommon)
#target_link_libraries(test_people /usr/lib/libglut.so)
#find_package(VTK REQUIRED)
#include_directories(${VTK_INCLUDE_DIRS})
#target_link_libraries(test_people ${PCL17_LIBRARIES} libvtkCommon.so libvtkFiltering.so libvtkRendering.so)
#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})
