#include <ceres/ceres.h>
#include <ceres/rotation.h>

//c++ lib
#include <cmath>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <fstream>

//ros lib
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

//pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//local lib
// #include "lidar.h"
// #include "odomEstimationClass.h"
#include <jh_mapping/TERRA.h>
#include <jh_mapping/lidarOptimization.h>
#include <jh_mapping/tic_toc.h>
// OdomEstimationClass odomEstimation;
std::mutex mutex_lock;
std::mutex mtx_path;

std::queue<sensor_msgs::PointCloud2ConstPtr> LiDARscanBuf;

pcl::KdTreeFLANN<PointType>::Ptr index_tree;
std::vector<Eigen::Isometry3d> pose_list;
std::vector<double> pose_time_list;

std::shared_ptr<TERRA> TERRA_local_map;
// TERRA TERRA_local_map;

ros::Publisher pubLaserOdometry;
ros::Publisher pubball_scan;
ros::Publisher pubtfscan;
ros::Publisher pubball_local_map;
ros::Publisher pubball_global_map;
ros::Publisher pubpointcloud_feature;
ros::Publisher pubpointcloud_indice;
ros::Publisher pubpointcloud_regflag;

ros::Publisher pubpointcloud_normal;
ros::Publisher pubpointcloud_mineigenvalue;
ros::Publisher pubros_feature;
ros::Publisher pubfeature_global;
ros::Publisher pubfeature_plane;
// Eigen::Isometry3d odom = Eigen::Isometry3d::Identity();
// Eigen::Isometry3d last_odom = Eigen::Isometry3d::Identity();
Eigen::Isometry3d odom_prediction = Eigen::Isometry3d::Identity();

pcl::PointCloud<PointType>::Ptr ball_global_map(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr tfscan_base(new pcl::PointCloud<PointType>());

int param_iter = 0;
int base_gap = 5;
int downsample_rate = 10;
double min_dis = 2;
int param_ball_num = 50000;

bool is_odom_inited = false;
double total_time =0;
int total_frame=0;
int plane_thres = 10;
int area_point_thres = 10;
double inlier_thres = 1;
double occlusion_thres = 0.4;
double ball_height = 0;
double ransac_thres = 0.01;

std::ofstream outfile;
std::string odom_path;

void LiDARscanHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    mutex_lock.lock();
    LiDARscanBuf.push(laserCloudMsg);
    mutex_lock.unlock();
}


void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg)
{
    double timestamp = msg->header.stamp.toSec();
    Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();
    std::cout<<"receive odom:"<<msg->pose.pose.position.x<<","<<msg->pose.pose.position.y<<","<<msg->pose.pose.position.z<<
                " "<<msg->pose.pose.orientation.w<<","<<msg->pose.pose.orientation.x<<","<<msg->pose.pose.orientation.y<<","<<msg->pose.pose.orientation.z <<std::endl;
    current_pose.rotate(Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z));  
    current_pose.pretranslate(Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));
    mtx_path.lock();
    pose_list.push_back(orthogonalize(current_pose));
    pose_time_list.push_back(timestamp);
    mtx_path.unlock();
}
void move_point(PointType const *const pi, PointType *const po, Eigen::Quaterniond q, Eigen::Vector3d t)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q * point_curr + t;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    // po->r = pi->r;
    // po->g = pi->g;
    // po->b = pi->b;
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}

std::mutex mtx_odom; // Global mutex lock
std::vector<Eigen::Vector3d> curr_point_list;
std::vector<Eigen::Vector3d> norm_list;
std::vector<double> negative_OA_dot_norm_list;
std::vector<double> weight_list;

void add_to_vectors(Eigen::Vector3d curr_point, Eigen::Vector3d norm, double negative_OA_dot_norm, double weight = 1.0) {
    std::lock_guard<std::mutex> lock(mtx_odom); // Lock the mutex
    curr_point_list.push_back(curr_point);
    norm_list.push_back(norm);
    negative_OA_dot_norm_list.push_back(negative_OA_dot_norm);
    weight_list.push_back(weight);
}





void odom_estimation(){
    std::cout<<"odom_estimation"<<std::endl;
    int pose_i = 0;
    pcl::PointCloud<PointType>::Ptr scan_raw(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr scan(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr scan_base(new pcl::PointCloud<PointType>());
    while(1){
        if(!LiDARscanBuf.empty()){
            double timestamp = LiDARscanBuf.front()->header.stamp.toSec();
            while(pose_time_list.empty() || timestamp > pose_time_list[pose_time_list.size()-1]){
                std::chrono::milliseconds dura(500);
                std::this_thread::sleep_for(dura); 
            }
            while(timestamp != pose_time_list[pose_i] && pose_i < (int)pose_time_list.size())
                pose_i++;
            mtx_path.lock();
            std::cout<<"pose_i:"<<pose_i<<"pose_time_list.size():"<<pose_time_list.size()<<std::endl;
            if(timestamp != pose_time_list[pose_i])
                ROS_WARN("localmap time not match with pose time");

            while(!LiDARscanBuf.empty() && LiDARscanBuf.front()->header.stamp.toSec() < timestamp)
                LiDARscanBuf.pop();
            pcl::PointCloud<PointType>::Ptr scan_raw(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*LiDARscanBuf.front(), *scan_raw);
            LiDARscanBuf.pop();
            mtx_path.unlock();

            pcl::PointCloud<PointType>::Ptr scan(new pcl::PointCloud<PointType>());
            for(int i = 0; i < (int)scan_raw->points.size(); i++){
                scan_raw->points[i].z -= ball_height; //handle z
                if(pcl::isFinite(scan_raw->points[i]) && (scan_raw->points[i].x*scan_raw->points[i].x + scan_raw->points[i].y*scan_raw->points[i].y + scan_raw->points[i].z*scan_raw->points[i].z > min_dis*min_dis)){
                    PointType pi = CorrectKITTIScan_kiss(scan_raw->points[i]);
                    pi.z = pi.z - ball_height;  //handle z
                    scan->points.push_back(pi);
                    scan_raw->points[i] = pi;
                }
            }
            Eigen::Isometry3d pose = pose_list[pose_i];



            TicToc t_start;
            if(total_frame == 0){
                Eigen::Isometry3d base_pose = pose;
                // base_pose.translation().z() = ball_height;
                TERRA_local_map = std::make_shared<TERRA>(TransformCloud(scan, base_pose), index_tree, base_pose, timestamp);
                TERRA_local_map->calculate_feature_for(downsample_rate, false);
                // TERRA_local_map->calculate_normal();
                // TERRA_local_map->calculate_normal_and_feature(plane_thres, area_point_thres, false);
                // TERRA_local_map->evaluate_degeneracy();
                // TERRA_local_map->pub_ros_feature(pubros_feature, "map");

                *ball_global_map += * TERRA_local_map->get_feature_ball_global();


            }
            // TERRA_local_map->draw_pointcloud_regist_flag(pubpointcloud_regflag, timestamp, "map");
            // TERRA_local_map->draw_pointcloud_feature(pubpointcloud_feature, timestamp, "map");
            // TERRA_local_map->draw_feature_global(pubfeature_global, timestamp, "map");

            // TERRA_local_map->draw_pointcloud_indice(pubpointcloud_indice, timestamp, "map");
            // TERRA_local_map->draw_pointcloud_normal(pubpointcloud_normal, timestamp, "map");
            // TERRA_local_map->draw_pointcloud_mineigenvalue(pubpointcloud_mineigenvalue, timestamp, "map");
            // TERRA_local_map->draw_feature_plane(pubfeature_plane, timestamp, "map");


            TERRA_local_map->clean_regist_flag();
            TERRA_local_map->evaluate_degeneracy();

            

            if((TERRA_local_map->get_base_pose().translation() - pose.translation()).norm() > base_gap && total_frame > 0){
                // TERRA_local_map->draw_pointcloud_indice(pubpointcloud_indice, timestamp, "map");

                TERRA_local_map->calculate_normal_and_feature3(plane_thres, area_point_thres, ransac_thres, false);
                // TERRA_local_map->draw_pointcloud_normal(pubpointcloud_normal, timestamp, "map");

                // TERRA_frame->draw_feature_plane(pubfeature_plane, timestamp, "map");

                TERRA_local_map->pub_ros_feature(pubros_feature, "map");
                *ball_global_map += * TERRA_local_map->get_feature_ball_global();
                // ROS_WARN("+++++++++++++++ new base++++++++++++++");

                TERRA_local_map = std::make_shared<TERRA>(scan_raw, index_tree, pose, timestamp);
                TERRA_local_map->calculate_feature_for(downsample_rate, false);
                // TERRA_local_map->calculate_normal();
                // TERRA_local_map->calculate_normal_and_feature(plane_thres, area_point_thres, false);
                // TERRA_local_map->evaluate_degeneracy();
                
                // TERRA_local_map->draw_ball(pubball_scan, timestamp, "map");
                // TERRA_local_map->draw_pointcloud_feature(pubpointcloud_feature, timestamp, "map");
                // TERRA_local_map->draw_pointcloud_indice(pubpointcloud_indice, timestamp, "map");
                // TERRA_local_map->draw_pointcloud_normal(pubpointcloud_normal, timestamp, "map");


            }
            else{ // update

                tfscan_base = TransformCloud(scan, TERRA_local_map->get_base_pose().inverse() * pose);
                std::shared_ptr<TERRA> TERRA_frame = std::make_shared<TERRA>(tfscan_base, index_tree, pose, timestamp);
                TERRA_frame->calculate_feature_for(downsample_rate, false);


                // TERRA_frame->calculate_normal_and_feature(plane_thres, area_point_thres, false);

                // TERRA_frame->draw_pointcloud_indice(pubpointcloud_indice, timestamp, "map");
                // TERRA_frame->draw_pointcloud_normal(pubpointcloud_normal, timestamp, "map");
                // TERRA_frame->draw_pointcloud_mineigenvalue(pubpointcloud_mineigenvalue, timestamp, "map");
                // TERRA_frame->draw_feature_plane(pubfeature_plane, timestamp, "map");

                int occluded_num = TERRA_local_map->update3(TERRA_frame, plane_thres);
                // TERRA_local_map->evaluate_degeneracy();

                TERRA_local_map->draw_ball(pubball_scan, timestamp, "map");

                std::cout<<"valid_num: "<<TERRA_local_map->get_valid_feature_num()<<std::endl;
                if(occluded_num > TERRA_local_map->get_valid_feature_num() * occlusion_thres ){
                    ROS_WARN("+++++++++++++++ new base for occlusion +++++++++++++++");
                    TERRA_local_map->pub_ros_feature(pubros_feature, "map");
                    *ball_global_map += * TERRA_local_map->get_feature_ball_global();
                    TERRA_local_map = std::make_shared<TERRA>(scan_raw, index_tree, pose, timestamp);
                    TERRA_local_map->calculate_feature_for(downsample_rate, false);

                    // TERRA_local_map->calculate_normal_and_feature(plane_thres, area_point_thres, false);
                    // TERRA_local_map->evaluate_degeneracy();

                }
                // if(TERRA_local_map->new_num > TERRA_local_map->get_valid_feature_num() / 2 ){
                //     ROS_WARN("+++++++++++++++ new base for new +++++++++++++++");
                //     TERRA_local_map->pub_ros_feature(pubros_feature, "map");
                //     *ball_global_map += * TERRA_local_map->get_feature_ball_global();
                //     TERRA_local_map = std::make_shared<TERRA>(scan_raw, index_tree, pose, timestamp);
                //     TERRA_local_map->calculate_normal_and_feature(plane_thres, area_point_thres, false);
                //     TERRA_local_map->evaluate_degeneracy();

                // }

            }

            Eigen::Quaterniond q_current(pose.rotation());
	        Eigen::Vector3d t_current = pose.translation();
            
            std::cout<<"time:"<<t_start.toc()<<"ms"<<std::endl;
            std::cout<<"frame "<<total_frame<<" "<<"t:"<<pose.translation().transpose()
                                                <<" "<<"q:"<<q_current.x()<<" "<<q_current.y()<<" "<<q_current.z()<<" "<<q_current.w()<<std::endl<<std::endl;
            
            Eigen::Matrix4d rm = pose.matrix();
            outfile<<rm(0,0)<<" "<<rm(0,1)<<" "<<rm(0,2)<<" "<<rm(0,3)<<" "
                   <<rm(1,0)<<" "<<rm(1,1)<<" "<<rm(1,2)<<" "<<rm(1,3)<<" "
                   <<rm(2,0)<<" "<<rm(2,1)<<" "<<rm(2,2)<<" "<<rm(2,3)<<std::endl;
            //pub odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "map";
            laserOdometry.child_frame_id = "base_link";
            laserOdometry.header.stamp = ros::Time().fromSec(timestamp);
            laserOdometry.pose.pose.orientation.x = q_current.x();
            laserOdometry.pose.pose.orientation.y = q_current.y();
            laserOdometry.pose.pose.orientation.z = q_current.z();
            laserOdometry.pose.pose.orientation.w = q_current.w();
            laserOdometry.pose.pose.position.x = t_current.x();
            laserOdometry.pose.pose.position.y = t_current.y();
            laserOdometry.pose.pose.position.z = t_current.z();
            pubLaserOdometry.publish(laserOdometry);

            // pub global ball map
            sensor_msgs::PointCloud2 cloudtempmsg;
            pcl::toROSMsg(*ball_global_map, cloudtempmsg);
            cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
            cloudtempmsg.header.frame_id = "map";
            pubball_global_map.publish(cloudtempmsg);

            // pub pointcloud map
            pcl::PointCloud<PointType>::Ptr tfscan = TransformCloud(scan_raw, pose);
            pcl::toROSMsg(*tfscan, cloudtempmsg); // tfscan_base, tfscan
            cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
            cloudtempmsg.header.frame_id = "map";
            pubtfscan.publish(cloudtempmsg);

            total_frame += 1;

        }
        //sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh("~"); 

    std::cout<<"main function"<<std::endl;


    std::string topic = "/rslidar_points";
    std::string odomtopic = "/rslidar_points";

    nh.getParam("param_ball_num", param_ball_num); 
    nh.getParam("base_gap", base_gap); 
    nh.getParam("param_iter", param_iter);
    nh.getParam("downsample_rate", downsample_rate);
    nh.getParam("odom_path", odom_path);
    nh.getParam("plane_thres", plane_thres);
    nh.getParam("area_point_thres", area_point_thres);
    nh.getParam("inlier_thres", inlier_thres);
    nh.getParam("occlusion_thres", occlusion_thres);
    nh.getParam("ball_height", ball_height);
    nh.getParam("min_dis", min_dis);
    nh.getParam("ransac_thres", ransac_thres);


    nh.getParam("odomtopic", odomtopic);
    nh.getParam("topic", topic);



    

    outfile.open(odom_path, ios::out | ios::trunc);

    
    std::cout<<"input topic:"<< topic<<std::endl;
    std::cout<<"input odom topic:"<< odomtopic<<std::endl;


    index_tree =  pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    index_tree->setInputCloud(generate_ball(param_ball_num));


    ros::Subscriber subLiDARscan = nh.subscribe<sensor_msgs::PointCloud2>(topic, 1000, LiDARscanHandler);
    ros::Subscriber subodom = nh.subscribe<nav_msgs::Odometry>(odomtopic, 1000, OdomHandler);

    pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/TERRAodom", 100);
    pubball_scan = nh.advertise<sensor_msgs::PointCloud2>("/ball_scan", 1000);
    pubtfscan = nh.advertise<sensor_msgs::PointCloud2>("/tfscan", 1000);
    pubball_local_map = nh.advertise<sensor_msgs::PointCloud2>("/ball_local_map", 1000);
    pubball_global_map = nh.advertise<sensor_msgs::PointCloud2>("/ball_global_map", 1000);
    pubpointcloud_feature = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_feature", 1000); 
    pubfeature_global = nh.advertise<sensor_msgs::PointCloud2>("/feature_global", 1000); 
    pubpointcloud_indice = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_indice", 1000); 
    pubpointcloud_regflag = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_regflag", 1000); 
    pubpointcloud_normal = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_normal", 1000); 
    pubpointcloud_mineigenvalue = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_mineigenvalue", 1000); 
    pubros_feature = nh.advertise<sensor_msgs::PointCloud2>("/ros_feature", 1000);
    pubfeature_plane = nh.advertise<sensor_msgs::PointCloud2>("/feature_plane", 1000);
    std::thread odom_estimation_process{odom_estimation};
    ros::spin();

    return 0;
}

