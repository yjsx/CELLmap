#include <ros/ros.h>  
#include <nav_msgs/Odometry.h>
#include <eigen3/Eigen/Dense>  
#include <iostream>  
#include <fstream>  
#include <tf/transform_broadcaster.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <chrono>
#include <thread>
struct PoseWithTimestamp {
    double timestamp;
    Eigen::Matrix<float, 4, 4> pose;
};

struct RosbagMessage {
    ros::Time timestamp;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> pointcloud;
    boost::shared_ptr<nav_msgs::Odometry const> odom;
};

// Function to read TUM format pose data
std::vector<PoseWithTimestamp> loadPoses(const std::string &path) {
    std::vector<PoseWithTimestamp> poses;
    std::ifstream input(path);
    if (!input.good()) {
        std::cerr << "Could not read pose file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (getline(input, line)) {
        std::stringstream ss(line);
        PoseWithTimestamp pose_data;
        
        // Read timestamp and position
        ss >> pose_data.timestamp;
        float x, y, z, qx, qy, qz, qw;
        ss >> x >> y >> z >> qx >> qy >> qz >> qw;

        // Convert quaternion to rotation matrix
        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Matrix<float, 4, 4> pose = Eigen::Matrix<float, 4, 4>::Identity();
        pose.block<3, 3>(0, 0) = q.toRotationMatrix();
        pose(0, 3) = x;
        pose(1, 3) = y;
        pose(2, 3) = z;

        pose_data.pose = pose;
        poses.push_back(pose_data);
    }
    input.close();
    return poses;
}


int main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "cloud_reader");
    ros::NodeHandle nh("~");

    // Publishers
    ros::Publisher pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/jh_cloud", 100);
    int rate_in = 10;
    // Get parameters
    std::string  rosbag_path, pointcloud_topic;
    nh.param<std::string>("rosbag_path", rosbag_path, "");
    nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/pointcloud");
    nh.param<int>("rate_in", rate_in, 10);
    
    std::this_thread::sleep_for(std::chrono::seconds(5));


    if (rosbag_path.empty()) {
        ROS_ERROR("Missing required parameters: pose_file_path and rosbag_path");
        return -1;
    }


    // Open rosbag
    rosbag::Bag bag;
    try {
        bag.open(rosbag_path, rosbag::bagmode::Read);
    } catch (rosbag::BagException& e) {
        ROS_ERROR("Failed to open rosbag: %s", e.what());
        return -1;
    }

    // Create view for pointcloud topic
    rosbag::View view(bag, rosbag::TopicQuery(pointcloud_topic));
    ros::Rate rate(rate_in);  // 30 Hz

    // Create message queue
    std::vector<RosbagMessage> messages;
    for (const rosbag::MessageInstance& m : view) {
        RosbagMessage msg;
        msg.timestamp = m.getTime();
        
        if (m.getTopic() == pointcloud_topic) {
            msg.pointcloud = m.instantiate<sensor_msgs::PointCloud2>();
            messages.push_back(msg);
            sensor_msgs::PointCloud2 cloud = *msg.pointcloud;
            cloud.header.stamp = msg.timestamp;
            cloud_pub.publish(cloud);
            rate.sleep();
        }
    }

    
    bag.close();

    return 0;  
}