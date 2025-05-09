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
    Eigen::Matrix<float, 7, 1> pose;
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

        // Store the position and quaternion directly into the pose vector
        pose_data.pose << x, y, z, qx, qy, qz, qw;
        poses.push_back(pose_data);
    }
    input.close();
    return poses;
}


int main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "pose_reader");
    ros::NodeHandle nh("~");

    // Publishers
    ros::Publisher pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/raw_cloud", 100);
    int rate_in = 10;
    int time_type = 0; // 0 for bag time, 1 for head time
    // Get parameters
    std::string pose_file_path, rosbag_path, pointcloud_topic;
    nh.param<std::string>("pose_file_path", pose_file_path, "");
    nh.param<std::string>("rosbag_path", rosbag_path, "");
    nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/pointcloud");
    nh.param<int>("rate_in", rate_in, 10);
    nh.param<int>("time_type", time_type, 0);


    std::this_thread::sleep_for(std::chrono::seconds(3));

    if (pose_file_path.empty() || rosbag_path.empty()) {
        ROS_ERROR("Missing required parameters: pose_file_path and rosbag_path");
        return -1;
    }

    // Load poses with timestamps
    std::vector<PoseWithTimestamp> poses = loadPoses(pose_file_path);

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

    size_t pose_idx = 0;
    size_t msg_idx = 0;
    ros::Rate rate(rate_in);  // 30 Hz
    const double sync_threshold = 0.01; // 10ms sync threshold
    std::cout<<"Start reading rosbag and publishing pose and pointcloud."<<std::endl;

    for (const rosbag::MessageInstance& m : view) {
        if( pose_idx >= poses.size() || ros::ok() == false)
            break;
        const auto& pose_data = poses[pose_idx];
        
        RosbagMessage msg;
        if (m.getTopic() == pointcloud_topic) {
            msg.pointcloud = m.instantiate<sensor_msgs::PointCloud2>();
            if(time_type == 0)
                msg.timestamp = m.getTime();
            else {
                msg.timestamp = msg.pointcloud->header.stamp;
            }

        
            double time_diff = fabs(pose_data.timestamp - msg.timestamp.toSec());
            if (time_diff < sync_threshold) {
                nav_msgs::Odometry ros_pose;
                ros_pose.header.stamp = ros::Time().fromSec(pose_data.timestamp);
                ros_pose.header.frame_id = "map";
                ros_pose.pose.pose.position.x = pose_data.pose(0);
                ros_pose.pose.pose.position.y = pose_data.pose(1);
                ros_pose.pose.pose.position.z = pose_data.pose(2);
                ros_pose.pose.pose.orientation.x = pose_data.pose(3);
                ros_pose.pose.pose.orientation.y = pose_data.pose(4);
                ros_pose.pose.pose.orientation.z = pose_data.pose(5);
                ros_pose.pose.pose.orientation.w = pose_data.pose(6);
                pose_pub.publish(ros_pose);
                sensor_msgs::PointCloud2 cloud = *msg.pointcloud;
                cloud.header.stamp = ros_pose.header.stamp;
                cloud_pub.publish(cloud);
                pose_idx++;
            }
            else{
                ROS_WARN("Time difference is too large: %f", time_diff);
            }
            rate.sleep();
        }
    }
    
    bag.close();

    return 0;  
}