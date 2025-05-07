#ifndef _TEST_H_
#define _TEST_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>


// #include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI PointType;

pcl::PointCloud<PointType>::Ptr  TransformCloud(pcl::PointCloud<PointType>::Ptr cloudin,    Eigen::Quaterniond q,  Eigen::Vector3d  t)
{
    pcl::PointCloud<PointType>::Ptr cloudout(new pcl::PointCloud<PointType> );
    PointType tmppoint;
    for(size_t i = 0;i < cloudin->points.size(); i++){
        Eigen::Vector3d point(cloudin->points[i].x, cloudin->points[i].y, cloudin->points[i].z);
        Eigen::Vector3d un_point =q * point +t;
        
        tmppoint.x = un_point.x();
        tmppoint.y = un_point.y();
        tmppoint.z = un_point.z();
        
        tmppoint.intensity = cloudin->points[i].intensity;
        
        // tmppoint.r = cloudin->points[i].r;
        // tmppoint.g = cloudin->points[i].g;
        // tmppoint.b = cloudin->points[i].b;
        cloudout->points.push_back(tmppoint);
    }
    return cloudout;
}

pcl::PointCloud<PointType>::Ptr  TransformCloud(pcl::PointCloud<PointType>::Ptr cloudin,  const Eigen::Isometry3d& pose)
{
    pcl::PointCloud<PointType>::Ptr cloudout(new pcl::PointCloud<PointType> );
    for(size_t i = 0;i < cloudin->points.size(); i++){
        PointType tmppoint;

        Eigen::Vector3d point(cloudin->points[i].x, cloudin->points[i].y, cloudin->points[i].z);
        Eigen::Vector3d tf_point = pose * point;
        
        tmppoint.x = tf_point.x();
        tmppoint.y = tf_point.y();
        tmppoint.z = tf_point.z();
        
        tmppoint.intensity = cloudin->points[i].intensity;
        
        // tmppoint.r = cloudin->points[i].r;
        // tmppoint.g = cloudin->points[i].g;
        // tmppoint.b = cloudin->points[i].b;
        cloudout->points.push_back(tmppoint);
    }
    return cloudout;
}

pcl::PointCloud<PointType>::Ptr generate_ball(int N) {
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);

    double phi = (std::sqrt(5.0) + 1.0) / 2.0;  // golden ratio
    double z_thresh1 = -0.60; //-0.3
    double z_thresh2 = -0.20; //-0.3

    for (int i = 0; i < N; ++i) {
        double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
        double radius = std::sqrt(1 - y*y);  // radius at y

        double theta = 2 * M_PI * i / phi;

        double x = radius * std::cos(theta);
        double z = radius * std::sin(theta);
        if( z >= z_thresh2 && z < 0.7 ){ ///-0.7~0.7
            PointType point;
            point.x = x;
            point.y = y;
            point.z = z;
            cloud->points.push_back(point);
        }
    }

    N = N * 2;

    for (int i = 0; i < N; ++i) {
        double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
        double radius = std::sqrt(1 - y*y);  // radius at y

        double theta = 2 * M_PI * i / phi;

        double x = radius * std::cos(theta);
        double z = radius * std::sin(theta);
        if( z >= z_thresh1 && z < z_thresh2 ){ ///-0.7~0.7
            PointType point;
            point.x = x;
            point.y = y;
            point.z = z;
            cloud->points.push_back(point);
        }
    }

    N = N / 200 / 2;
    for (int i = 0; i < N; ++i) {
        double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
        double radius = std::sqrt(1 - y*y);  // radius at y

        double theta = 2 * M_PI * i / phi;

        double x = radius * std::cos(theta);
        double z = radius * std::sin(theta);
        if(z < z_thresh1){    ///<-0.7

            PointType point;
            point.x = x;
            point.y = y;
            point.z = z;
            // std::cout<<"add ground point:"<<point.x<<" "<<point.y<<" "<<point.z<<std::endl;
            cloud->points.push_back(point);
        }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    return cloud;
}

// pcl::PointCloud<PointType>::Ptr generate_ball(int N) { // for visualization
//     pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);

//     double phi = (std::sqrt(5.0) + 1.0) / 2.0;  // golden ratio
//     double z_thresh1 = -0.30; //-0.3
//     double z_thresh2 = -0.30; //-0.3

//     for (int i = 0; i < N; ++i) {
//         double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
//         double radius = std::sqrt(1 - y*y);  // radius at y

//         double theta = 2 * M_PI * i / phi;

//         double x = radius * std::cos(theta);
//         double z = radius * std::sin(theta);
//         if( z >= z_thresh2 && z < 0.7 ){ ///-0.7~0.7
//             PointType point;
//             point.x = x;
//             point.y = y;
//             point.z = z;
//             cloud->points.push_back(point);
//         }
//     }

//     N = N * 2;

//     for (int i = 0; i < N; ++i) {
//         double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
//         double radius = std::sqrt(1 - y*y);  // radius at y

//         double theta = 2 * M_PI * i / phi;

//         double x = radius * std::cos(theta);
//         double z = radius * std::sin(theta);
//         if( z >= z_thresh1 && z < z_thresh2 ){ ///-0.7~0.7
//             PointType point;
//             point.x = x;
//             point.y = y;
//             point.z = z;
//             cloud->points.push_back(point);
//         }
//     }

//     N = N / 200 / 2;
//     for (int i = 0; i < N; ++i) {
//         double y = 1 - ((i / static_cast<double>(N-1)) * 2);  // y goes from 1 to 0
//         double radius = std::sqrt(1 - y*y);  // radius at y

//         double theta = 2 * M_PI * i / phi;

//         double x = radius * std::cos(theta);
//         double z = radius * std::sin(theta);
//         if(z < z_thresh1){    ///<-0.7

//             PointType point;
//             point.x = x;
//             point.y = y;
//             point.z = z;
//             std::cout<<"add ground point:"<<point.x<<" "<<point.y<<" "<<point.z<<std::endl;
//             cloud->points.push_back(point);
//         }
//     }
//     cloud->width = cloud->points.size();
//     cloud->height = 1;
//     return cloud;
// }

Eigen::Isometry3d qt2Isometry3d(Eigen::Quaterniond q, Eigen::Vector3d t){	
	Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
	mat.rotate(q.matrix());
	mat.pretranslate(t);
	return mat;
}

Eigen::Isometry3d orthogonalize(const Eigen::Isometry3d& transform) {
    Eigen::Matrix3d rotation = transform.rotation();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d orthogonalized_rotation = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Isometry3d result = transform;
    result.linear() = orthogonalized_rotation;
    return result;
}



PointType CorrectKITTIScan_kiss(PointType pt_in) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.195 * M_PI) / 180.0;
    Eigen::Vector3d pt(pt_in.x, pt_in.y, pt_in.z);
    const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
    pt = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
    PointType pt_out;
    pt_out.x = pt.x();
    pt_out.y = pt.y();
    pt_out.z = pt.z();
    pt_out.intensity = pt_in.intensity;
    return pt_out;
}

PointType CorrectKITTIScan_pfilter(PointType pi) {
    double range = sqrt(pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
    double calib_vertical_angle = 0.15* PI / 180.0;
    double vertical_angle = asin(pi.z / range) + calib_vertical_angle;
    double horizon_angle = atan2(pi.y, pi.x);
    pi.z = range * sin(vertical_angle);
    double project_len = range * cos(vertical_angle);
    pi.x = project_len * cos(horizon_angle);
    pi.y = project_len * sin(horizon_angle);
    return pi;
}

#endif
