#include <ros/ros.h>  
#include <nav_msgs/Odometry.h>
#include <eigen3/Eigen/Dense>  
#include <iostream>  
#include <fstream>  
#include <tf/transform_broadcaster.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

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
    ros::init(argc, argv, "pose_reader");
    ros::NodeHandle nh("~");

    // Publishers
    ros::Publisher pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/jh_cloud", 100);
    int rate_in = 10;
    // Get parameters
    std::string pose_file_path, rosbag_path, pointcloud_topic;
    nh.param<std::string>("pose_file_path", pose_file_path, "");
    nh.param<std::string>("rosbag_path", rosbag_path, "");
    nh.param<std::string>("pointcloud_topic", pointcloud_topic, "/pointcloud");
    nh.param<int>("rate_in", rate_in, 10);


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

    // Create message queue
    std::vector<RosbagMessage> messages;
    for (const rosbag::MessageInstance& m : view) {
        RosbagMessage msg;
        msg.timestamp = m.getTime();
        
        if (m.getTopic() == pointcloud_topic) {
            msg.pointcloud = m.instantiate<sensor_msgs::PointCloud2>();
        }
        messages.push_back(msg);
    }

    // Sort messages by timestamp
    std::sort(messages.begin(), messages.end(), 
        [](const RosbagMessage& a, const RosbagMessage& b) {
            return a.timestamp < b.timestamp;
        });

    // Main loop
    size_t pose_idx = 0;
    size_t msg_idx = 0;
    ros::Rate rate(rate_in);  // 30 Hz
    const double sync_threshold = 0.01; // 10ms sync threshold
    std::cout<<"Start reading rosbag and publishing pose and pointcloud."<<std::endl;
    while (ros::ok() && pose_idx < poses.size() && msg_idx < messages.size()) {
        const auto& pose_data = poses[pose_idx];
        const auto& msg = messages[msg_idx];
        double time_diff = fabs(pose_data.timestamp - msg.timestamp.toSec());

        // Find synchronized messages
        if (time_diff < sync_threshold) {
            // Publish odometry
            nav_msgs::Odometry ros_pose;
            ros_pose.header.stamp = ros::Time().fromSec(pose_data.timestamp);
            ros_pose.header.frame_id = "map";
            ros_pose.pose.pose.position.x = pose_data.pose(0, 3);
            ros_pose.pose.pose.position.y = pose_data.pose(1, 3);
            ros_pose.pose.pose.position.z = pose_data.pose(2, 3);
            Eigen::Quaternionf q(pose_data.pose.block<3, 3>(0, 0));
            ros_pose.pose.pose.orientation.x = q.x();
            ros_pose.pose.pose.orientation.y = q.y();
            ros_pose.pose.pose.orientation.z = q.z();
            ros_pose.pose.pose.orientation.w = q.w();
            pose_pub.publish(ros_pose);

            // Publish synchronized pointcloud
            if (msg.pointcloud) {
                sensor_msgs::PointCloud2 cloud = *msg.pointcloud;
                cloud.header.stamp = ros_pose.header.stamp;
                cloud_pub.publish(cloud);
            }

            // Move both indices forward
            pose_idx++;
            msg_idx++;
        }
        // Move the earlier timestamp forward
        else if (pose_data.timestamp < msg.timestamp.toSec()) {
            pose_idx++;
        } else {
            msg_idx++;
        }

        rate.sleep();
    }

    bag.close();

    return 0;  
}