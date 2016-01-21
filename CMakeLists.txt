cmake_minimum_required(VERSION 2.8.3)
project(people_detection)
find_package(cmake_modules REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  geometry_msgs
#  VTK
# pcl
  pcl_ros
  pcl_conversions
  roscpp
  sensor_msgs
  shape_msgs
  std_msgs
  tf
  visualization_msgs
  message_generation
  roslib
)

add_message_files(
    FILES
    PersonObject.msg
    PersonObjectArray.msg
 )

add_service_files(
    FILES
    ClearPeopleTracker.srv
)
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  sensor_msgs
  )

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS geometry_msgs pcl_ros pcl_conversions roscpp sensor_msgs shape_msgs std_msgs tf visualization_msgs
#  DEPENDS system_lib
)

###########
## Build ##
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(include ${catkin_INCLUDE_DIRS})

## Declare a cpp library
# add_library(people_detection
#   src/${PROJECT_NAME}/people_detection.cpp
# )

## Declare a cpp executable
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
#add_executable(people_detection src/people_detection.cpp)
#add_executable(people_detection_node src/people_detection_node_temp.cpp)
add_executable(people_visualize src/people_detection_visualization.cpp)
add_executable(people_detection_node src/people_detection_node.cpp src/PeopleDetector.cpp src/PeopleTracker.cpp)
add_executable(people_detection_node_temp src/people_detection_node_temp.cpp)


#target_link_libraries(people_detection libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkGraphics.so)
target_link_libraries(people_detection_node libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkGraphics.so)
target_link_libraries(people_visualize libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkGraphics.so)
target_link_libraries(people_detection_node_temp libvtkCommon.so libvtkFiltering.so libvtkRendering.so libvtkGraphics.so)

#target_link_libraries(people_detection ${pcl_ros_LIBRARIES})
target_link_libraries(people_detection_node ${pcl_ros_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(people_visualize ${pcl_ros_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(people_detection_node_temp ${pcl_ros_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})



#add_executable(people_detection_original src/people_detection_modify.cpp)
#target_link_libraries(people_detection_original libvtkCommon.so libvtkFiltering.so libvtkRendering.so)
