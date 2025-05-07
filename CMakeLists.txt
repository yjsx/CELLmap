cmake_minimum_required(VERSION 3.0.2)
project(CELLmap)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosbag
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  tf
  eigen_conversions
  sa_msgs
)

find_package(Eigen3)

if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()
find_package(PCL REQUIRED)
find_package(Ceres 2.0.0 REQUIRED)
find_package(OpenMP)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs sa_msgs
  DEPENDS EIGEN3 PCL Ceres 
  INCLUDE_DIRS include
)



# add_executable(record_kitti_tra src/record_kitti_tra.cpp)
# target_link_libraries(record_kitti_tra ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES})

add_executable(pub_pose_from_file src/pub_pose_from_file.cpp)
target_link_libraries(pub_pose_from_file ${catkin_LIBRARIES})

add_executable(pub_cloud_from_bag src/pub_cloud_from_bag.cpp)
target_link_libraries(pub_cloud_from_bag ${catkin_LIBRARIES})

add_executable(jh_pointcloud_to_cell_node src/pointcloud_to_cell.cpp src/lidarOptimization.cpp)
target_link_libraries(jh_pointcloud_to_cell_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(jh_backend_node src/backend.cpp src/lidarOptimization.cpp)
target_link_libraries(jh_backend_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} gtsam OpenMP::OpenMP_CXX)

add_executable(read_gnss_from_rosbag src/read_gnss_from_rosbag.cpp)
target_link_libraries(read_gnss_from_rosbag ${catkin_LIBRARIES})
