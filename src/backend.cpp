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
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
//pcl lib
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//local lib
// #include "lidar.h"
// #include "odomEstimationClass.h"
#include <cellmap/CELL.h>
#include <cellmap/lidarOptimization.h>
#include <cellmap/tic_toc.h>




// gtsam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
// OdomEstimationClass odomEstimation;

std::mutex mutex_gtsam;
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
gtsam::PriorFactor<gtsam::Pose3> prior_factor;

std::mutex mutex_lock;
std::queue<sensor_msgs::PointCloud2ConstPtr> LiDARscanBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> LocalmapBuf;
std::queue<nav_msgs::OdometryConstPtr> OdomBuf;


pcl::KdTreeFLANN<PointType>::Ptr index_tree;
pcl::KdTreeFLANN<PointType>::Ptr pose_tree;

std::vector<int> base_id_list;
std::vector<double> pose_time_list;
std::vector<Eigen::Isometry3d> pose_list;
std::vector<Eigen::Isometry3d> opti_pose_list;
std::vector<pcl::PointCloud<PointType>::Ptr> local_map_list;

std::shared_ptr<CELL> CELL_last_local_map;
std::shared_ptr<CELL> CELL_current_local_map;


ros::Publisher pubbefore_path;
ros::Publisher pubafter_path;
ros::Publisher publoop_line;
ros::Publisher publoop_CELL;
ros::Publisher publoop_scan;




// Eigen::Isometry3d odom = Eigen::Isometry3d::Identity();
// Eigen::Isometry3d last_odom = Eigen::Isometry3d::Identity();
Eigen::Isometry3d odom_prediction = Eigen::Isometry3d::Identity();

pcl::PointCloud<PointType>::Ptr ball_global_map(new pcl::PointCloud<PointType>());

int param_iter = 0;
double inlier_thres = 1;
int downsample_rate = 10;
double min_dis = 0.5;
bool is_odom_inited = false;
double total_time =0;
int total_frame=0;
double ball_height = 0;
int FLAG_LOOP = 0;
int loop_line_id_count = 0;
std::ofstream outfile;
std::ofstream cell_outfile;
std::string odom_path;
std::string cell_odom_path;
std::string output_mode = "kitti";





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

std::mutex mtx_odom; // 全局互斥锁
std::vector<Eigen::Vector3d> curr_point_list;
std::vector<Eigen::Vector3d> norm_list;
std::vector<double> negative_OA_dot_norm_list;
std::vector<double> weight_list;

void add_to_vectors(Eigen::Vector3d curr_point, Eigen::Vector3d norm, double negative_OA_dot_norm, double weight = 1.0) {
    std::lock_guard<std::mutex> lock(mtx_odom); // 锁定互斥锁
    curr_point_list.push_back(curr_point);
    norm_list.push_back(norm);
    negative_OA_dot_norm_list.push_back(negative_OA_dot_norm);
    weight_list.push_back(weight);
}


std::mutex mtx_path; // 全局互斥锁

void  publish_path(std::vector<Eigen::Isometry3d> pose_list, ros::Publisher pub, double timestamp = ros::Time::now().toSec()){
    nav_msgs::Path globalPath;
    globalPath.header.frame_id = "map";
    globalPath.header.stamp = ros::Time().fromSec(timestamp);

    mtx_path.lock();
    for(int i = 0; i < (int)pose_list.size(); i++){
        geometry_msgs::PoseStamped pose_stamped;
        Eigen::Quaterniond q_current(pose_list[i].rotation());
        Eigen::Vector3d t_current = pose_list[i].translation();
        pose_stamped.header.stamp = ros::Time().fromSec(timestamp);
        pose_stamped.header.frame_id = globalPath.header.frame_id;
        pose_stamped.pose.position.x = t_current.x();
        pose_stamped.pose.position.y = t_current.y();
        pose_stamped.pose.position.z = t_current.z();
        pose_stamped.pose.orientation.x = q_current.x();
        pose_stamped.pose.orientation.y = q_current.y();
        pose_stamped.pose.orientation.z = q_current.z();
        pose_stamped.pose.orientation.w = q_current.w();

        globalPath.poses.push_back(pose_stamped);
    }
    mtx_path.unlock();

    pub.publish(globalPath);
}


void pulish_loop_line(Eigen::Isometry3d start, Eigen::Isometry3d end, ros::Publisher pub, double timestamp, std::string color){
    visualization_msgs::Marker line_strip;  
    line_strip.header.frame_id =  "map";  
    line_strip.header.stamp = ros::Time().fromSec(timestamp);
    line_strip.ns = "lines";  
    line_strip.action = visualization_msgs::Marker::ADD;  
    line_strip.pose.orientation.w = 1.0;  

    line_strip.id = loop_line_id_count;
    loop_line_id_count ++;  

    line_strip.type = visualization_msgs::Marker::LINE_STRIP;  

    // 设置线束的尺寸为 0.1 米  
    line_strip.scale.x = 1;  

    // 设置线束的颜色为红色  
    if(color == "b"){
        line_strip.color.b = 1.0;  
    }
    else if(color == "g"){
        line_strip.color.g = 1.0;  
    }
    else if(color == "o"){
        line_strip.color.r = 1.0;  
        line_strip.color.g = 0.5;  
    }   

    line_strip.color.a = 1.0;  

    // 设置起点和终点  
    geometry_msgs::Point p1;  
    p1.x = start.translation().x();  
    p1.y = start.translation().y();  
    p1.z = start.translation().z();  
    
    line_strip.points.push_back(p1);  

    geometry_msgs::Point p2;  
    p2.x = end.translation().x();  
    p2.y = end.translation().y();  
    p2.z = end.translation().z();  
    line_strip.points.push_back(p2);  

    pub.publish(line_strip); 
}

Eigen::Isometry3d tfbetweenCELL(std::shared_ptr<CELL> CELL_last_map, std::shared_ptr<CELL> CELL_current_map){
    ////////////////////////////////////////////////////////////////////////////

    Eigen::Isometry3d forward_pose =  CELL_last_map->get_base_pose().inverse() * CELL_current_map->get_base_pose();
    Eigen::Quaterniond q_forward(forward_pose.rotation());
    Eigen::Vector3d t_forward = forward_pose.translation();
    double parameters[7] = {q_forward.x(), q_forward.y(), q_forward.z(), q_forward.w(), t_forward.x(), t_forward.y(), t_forward.z()};
    Eigen::Map<Eigen::Quaterniond> q_predict = Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_predict = Eigen::Map<Eigen::Vector3d>(parameters+4);

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
    for (int iter = 0; iter < param_iter; iter++){
        int valid_num = 0;

        TicToc t_add;

        pcl::PointCloud<PointType>::Ptr scan_opti;
        pcl::PointCloud<PointType>::Ptr scan_base = CELL_current_map->get_cloud();

    
        // if(iter != 0)
        scan_opti = TransformCloud(scan_base, q_predict, t_predict);
        // else
        // scan_opti = scan_base;

        CELL CELL_opti_current(scan_opti, index_tree, Eigen::Isometry3d::Identity());
        CELL_opti_current.calculate_feature_for(downsample_rate, false);
        std::cout<<"feature time: "<<t_add.toc()<<"ms"<<std::endl;
        std::cout<<"CELL_last_map->get_valid_feature_num()"<<CELL_last_map->get_valid_feature_num()<<std::endl;
        std::cout<<"CELL_last_map->get_valid_normal_num()"<<CELL_last_map->get_valid_normal_num()<<std::endl;


        #pragma omp parallel for num_threads(16) 
        for (int i = 0; i < (int)scan_opti->points.size(); i++){
            ////// downsample
            if(i % downsample_rate!= 0)
                continue;
            //////

            int feature_id = CELL_opti_current.get_point_index(i);
            if((CELL_last_map->get_ballpoint_feature(feature_id) != 0)   )  
            //    (std::abs(CELL_local_map->get_ballpoint_feature(feature_id) - CELL_opti.get_ballpoint_feature(feature_id)) < high_threshs[iter < high_threshs.size() - 1? iter : high_threshs.size() - 1])
               {
                Eigen::Vector3d norm = CELL_last_map->get_ballpoint_normal(feature_id);
                if(norm.sum() == 0)
                    continue;
                
                // double negative_OA_dot_norm = 1 / norm.norm();
                double negative_OA_dot_norm = - CELL_last_map->get_ballpoint_feature(feature_id) *
                                                (index_tree->getInputCloud()->points[feature_id].x * norm.x() + 
                                                 index_tree->getInputCloud()->points[feature_id].y * norm.y() + 
                                                 index_tree->getInputCloud()->points[feature_id].z * norm.z());

         
                Eigen::Vector3d point_eigen(scan_opti->points[i].x, scan_opti->points[i].y, scan_opti->points[i].z);
                if (abs(norm.dot(point_eigen) + negative_OA_dot_norm) >  inlier_thres / (iter + 1))
                    continue;
                CELL_last_map->set_regist_flag(feature_id);
                Eigen::Vector3d curr_point(scan_base->points[i].x, scan_base->points[i].y, scan_base->points[i].z);
                double weight = abs(norm.dot(q_predict * CELL_current_map->get_ballpoint_normal(CELL_current_map->get_point_index(i))));
                // std::cout<<"forward i:"<<i<<" index:"<< CELL_current_map->get_point_index(i)<<" weight: "<<weight
                //         <<" normal:"<<CELL_current_map->get_ballpoint_normal(CELL_current_map->get_point_index(i))[0]<<std::endl;
                // if(weight < 0.95 )
                //     continue;
                add_to_vectors(curr_point, norm, negative_OA_dot_norm, weight);
            }
        }

        valid_num += curr_point_list.size();
        for (int i = 0; i < (int)curr_point_list.size(); i++){
            ceres::CostFunction *cost_function = new SurfForwardCostFunction(curr_point_list[i], norm_list[i], negative_OA_dot_norm_list[i], weight_list[i]);    
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
        if(curr_point_list.size()<200){
            ROS_WARN("not enough correct points %d", (int)curr_point_list.size());
            return Eigen::Isometry3d::Identity();
        }

        curr_point_list.clear();
        norm_list.clear();
        negative_OA_dot_norm_list.clear();
        weight_list.clear();
        // forward
        ////////////////////////////////////////////////////////////////////////////
        // back
        Eigen::Isometry3d back_pose =  CELL_current_map->get_base_pose().inverse() * CELL_last_map->get_base_pose();
        Eigen::Quaterniond q_back(back_pose.rotation());
        Eigen::Vector3d t_back = back_pose.translation();

        ///////follow todo

        scan_base = CELL_last_map->get_cloud();
        scan_opti = TransformCloud(scan_base, q_back, t_back);
        // else
        // scan_opti = scan_base;

        CELL CELL_opti_last(scan_opti, index_tree, Eigen::Isometry3d::Identity());
        CELL_opti_last.calculate_feature_for(downsample_rate, false);
        std::cout<<"feature time: "<<t_add.toc()<<"ms"<<std::endl;
        #pragma omp parallel for num_threads(16) 
        for (int i = 0; i < (int)scan_opti->points.size(); i++){
            ////// downsample
            if(i % downsample_rate!= 0)
                continue;
            //////

            int feature_id = CELL_opti_last.get_point_index(i);
            if((CELL_current_map->get_ballpoint_feature(feature_id) != 0)  )  
            //    (std::abs(CELL_local_map->get_ballpoint_feature(feature_id) - CELL_opti.get_ballpoint_feature(feature_id)) < high_threshs[iter < high_threshs.size() - 1? iter : high_threshs.size() - 1])
               {
                Eigen::Vector3d norm = CELL_current_map->get_ballpoint_normal(feature_id);
                if(norm.sum() == 0)
                    continue;
                
                // double negative_OA_dot_norm = 1 / norm.norm();
                double negative_OA_dot_norm = - CELL_current_map->get_ballpoint_feature(feature_id) *
                                                (index_tree->getInputCloud()->points[feature_id].x * norm.x() + 
                                                 index_tree->getInputCloud()->points[feature_id].y * norm.y() + 
                                                 index_tree->getInputCloud()->points[feature_id].z * norm.z());

                Eigen::Vector3d point_eigen(scan_opti->points[i].x, scan_opti->points[i].y, scan_opti->points[i].z);

                if (abs(norm.dot(point_eigen) + negative_OA_dot_norm) >  inlier_thres / (iter + 1))
                    continue;
                CELL_current_map->set_regist_flag(feature_id);
                Eigen::Vector3d curr_point(scan_base->points[i].x, scan_base->points[i].y, scan_base->points[i].z);
                // Eigen::Quaterniond q_w_curr(parameters[3], parameters[0], parameters[1], parameters[2]);
                // Eigen::Vector3d t_w_curr(parameters[4], parameters[5], parameters[6]);
                // Eigen::Quaterniond q_inverse = q_w_curr.conjugate();
                // Eigen::Vector3d t_inverse = q_inverse * (-t_w_curr);
                // Eigen::Vector3d point_w = q_inverse * curr_point + t_inverse;
                
                double weight = abs(norm.dot(q_back * CELL_last_map->get_ballpoint_normal(CELL_last_map->get_point_index(i))));
                // std::cout<<"back i:"<<i<<" index:"<< CELL_last_map->get_point_index(i)<<" weight: "<<weight
                //         <<" normal:"<<CELL_last_map->get_ballpoint_normal(CELL_last_map->get_point_index(i))[0]<<std::endl;

                // if(weight < 0.95 )
                //     continue;
                add_to_vectors(curr_point, norm, negative_OA_dot_norm, weight);

            }
        }

        valid_num += curr_point_list.size();
        for (int i = 0; i < (int)curr_point_list.size(); i++){
            ceres::CostFunction *cost_function = new SurfbackCostFunction(curr_point_list[i], norm_list[i], negative_OA_dot_norm_list[i], weight_list[i]);    
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
        if(curr_point_list.size()<200){
            ROS_WARN("not enough correct points %d", (int)curr_point_list.size());
            return Eigen::Isometry3d::Identity();
        }

        curr_point_list.clear();
        norm_list.clear();
        negative_OA_dot_norm_list.clear();
        weight_list.clear();
        //////////////////////////////////////////////////////////////////////////// back

        std::cout<<"t_add: "<<t_add.toc()<<"ms"<<std::endl;
        ROS_WARN("valid_num %d / %d", valid_num, (int)CELL_current_map->get_cloud()->points.size() + (int)CELL_last_map->get_cloud()->points.size());
        TicToc t_solve;

        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 1e-4;
        options.num_threads = std::thread::hardware_concurrency();
        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);
        std::cout<<"t_solve: "<<t_solve.toc()<<"ms"<<std::endl;

    }
    std::cout<<"before: "<<t_forward.x()<<" "<<t_forward.y()<<" "<<t_forward.z()<<std::endl;
    std::cout<<"after: "<<t_predict.x()<<" "<<t_predict.y()<<" "<<t_predict.z()<<std::endl;
    return qt2Isometry3d(q_predict, t_predict);
}

//2 for only use CELL but not scan
Eigen::Isometry3d tfbetweenCELL2(std::shared_ptr<CELL> CELL_last_map, std::shared_ptr<CELL> CELL_current_map){
    ////////////////////////////////////////////////////////////////////////////

    Eigen::Isometry3d forward_pose =  CELL_last_map->get_base_pose().inverse() * CELL_current_map->get_base_pose();
    Eigen::Quaterniond q_forward(forward_pose.rotation());
    Eigen::Vector3d t_forward = forward_pose.translation();
    double parameters[7] = {q_forward.x(), q_forward.y(), q_forward.z(), q_forward.w(), t_forward.x(), t_forward.y(), t_forward.z()};
    Eigen::Map<Eigen::Quaterniond> q_predict = Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_predict = Eigen::Map<Eigen::Vector3d>(parameters+4);

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    double t_add_sum = 0;
    double t_solve_sum = 0;

    problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
    for (int iter = 0; iter < param_iter; iter++){
        int valid_num = 0;

        TicToc t_add;

        pcl::PointCloud<PointType>::Ptr scan_opti;
        pcl::PointCloud<PointType>::Ptr scan_base = CELL_current_map->get_ballpoint_times_feature();

    
        // if(iter != 0)
        scan_opti = TransformCloud(scan_base, q_predict, t_predict);
        // else
        // scan_opti = scan_base;

        CELL CELL_opti_current(scan_opti, index_tree, Eigen::Isometry3d::Identity());
        CELL_opti_current.calculate_feature_for(1, false);
        std::cout<<"feature time: "<<t_add.toc()<<"ms"<<std::endl;
        #pragma omp parallel for num_threads(16) 
        for (int i = 0; i < (int)scan_opti->points.size(); i++){
            

            int feature_id = i;
            if((CELL_last_map->get_ballpoint_feature(feature_id) != 0) && 
               (CELL_opti_current.get_ballpoint_feature(feature_id) != 0) &&
               (std::abs(CELL_last_map->get_ballpoint_feature(feature_id) - CELL_opti_current.get_ballpoint_feature(feature_id)) > 0)  
            //    (std::abs(CELL_local_map->get_ballpoint_feature(feature_id) - CELL_opti.get_ballpoint_feature(feature_id)) < high_threshs[iter < high_threshs.size() - 1? iter : high_threshs.size() - 1])
               ){
                Eigen::Vector3d norm = CELL_last_map->get_ballpoint_normal(feature_id);
                if(norm.sum() == 0)
                    continue;
                
                // double negative_OA_dot_norm = 1 / norm.norm();
                double negative_OA_dot_norm = - CELL_last_map->get_ballpoint_feature(feature_id) *
                                                (index_tree->getInputCloud()->points[feature_id].x * norm.x() + 
                                                 index_tree->getInputCloud()->points[feature_id].y * norm.y() + 
                                                 index_tree->getInputCloud()->points[feature_id].z * norm.z());

         
                Eigen::Vector3d point_eigen(scan_opti->points[i].x, scan_opti->points[i].y, scan_opti->points[i].z);
                if (abs(norm.dot(point_eigen) + negative_OA_dot_norm) >  inlier_thres / (iter + 1))
                    continue;
                CELL_last_map->set_regist_flag(feature_id);
                Eigen::Vector3d curr_point(scan_base->points[i].x, scan_base->points[i].y, scan_base->points[i].z);
                double weight = abs(norm.dot(q_predict * CELL_current_map->get_ballpoint_normal(CELL_current_map->get_point_index(i))));
                // std::cout<<"weight: "<<weight<<std::endl;

   
          
                add_to_vectors(curr_point, norm, negative_OA_dot_norm, weight);
            }
        }

        valid_num += curr_point_list.size();
        for (int i = 0; i < (int)curr_point_list.size(); i++){
            ceres::CostFunction *cost_function = new SurfForwardCostFunction(curr_point_list[i], norm_list[i], negative_OA_dot_norm_list[i], weight_list[i]);    
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
        if(curr_point_list.size()<20){
            ROS_WARN("not enough correct points %d", (int)curr_point_list.size());
            return Eigen::Isometry3d::Identity();
        }

        curr_point_list.clear();
        norm_list.clear();
        negative_OA_dot_norm_list.clear();
        weight_list.clear();
        // forward
        ////////////////////////////////////////////////////////////////////////////
        // back
        Eigen::Isometry3d back_pose =  CELL_current_map->get_base_pose().inverse() * CELL_last_map->get_base_pose();
        Eigen::Quaterniond q_back(back_pose.rotation());
        Eigen::Vector3d t_back = back_pose.translation();

        ///////follow todo

        scan_base = CELL_last_map->get_ballpoint_times_feature();
        scan_opti = TransformCloud(scan_base, q_back, t_back);
        // else
        // scan_opti = scan_base;

        CELL CELL_opti_last(scan_opti, index_tree, Eigen::Isometry3d::Identity());
        CELL_opti_last.calculate_feature_for(downsample_rate, false);
        std::cout<<"feature time: "<<t_add.toc()<<"ms"<<std::endl;
        #pragma omp parallel for num_threads(16) 
        for (int i = 0; i < (int)scan_opti->points.size(); i++){
       
            int feature_id = i;
            if((CELL_current_map->get_ballpoint_feature(feature_id) != 0) && 
               (CELL_opti_last.get_ballpoint_feature(feature_id) != 0) &&
               (std::abs(CELL_current_map->get_ballpoint_feature(feature_id) - CELL_opti_last.get_ballpoint_feature(feature_id)) > 0)  
            //    (std::abs(CELL_local_map->get_ballpoint_feature(feature_id) - CELL_opti.get_ballpoint_feature(feature_id)) < high_threshs[iter < high_threshs.size() - 1? iter : high_threshs.size() - 1])
               ){
                Eigen::Vector3d norm = CELL_current_map->get_ballpoint_normal(feature_id);
                if(norm.sum() == 0)
                    continue;
                
                // double negative_OA_dot_norm = 1 / norm.norm();
                double negative_OA_dot_norm = - CELL_current_map->get_ballpoint_feature(feature_id) *
                                                (index_tree->getInputCloud()->points[feature_id].x * norm.x() + 
                                                 index_tree->getInputCloud()->points[feature_id].y * norm.y() + 
                                                 index_tree->getInputCloud()->points[feature_id].z * norm.z());

                Eigen::Vector3d point_eigen(scan_opti->points[i].x, scan_opti->points[i].y, scan_opti->points[i].z);

                if (abs(norm.dot(point_eigen) + negative_OA_dot_norm) >  inlier_thres / (iter + 1))
                    continue;
                CELL_current_map->set_regist_flag(feature_id);
                Eigen::Vector3d curr_point(scan_base->points[i].x, scan_base->points[i].y, scan_base->points[i].z);
                // Eigen::Quaterniond q_w_curr(parameters[3], parameters[0], parameters[1], parameters[2]);
                // Eigen::Vector3d t_w_curr(parameters[4], parameters[5], parameters[6]);
                // Eigen::Quaterniond q_inverse = q_w_curr.conjugate();
                // Eigen::Vector3d t_inverse = q_inverse * (-t_w_curr);
                // Eigen::Vector3d point_w = q_inverse * curr_point + t_inverse;
                
                double weight = abs(norm.dot(q_back * CELL_last_map->get_ballpoint_normal(CELL_last_map->get_point_index(i))));
                // std::cout<<"i:"<<i<<" index:"<< CELL_last_map->get_point_index(i)<<" weight: "<<weight
                //         <<" normal:"<<CELL_last_map->get_ballpoint_normal(CELL_last_map->get_point_index(i))<<std::endl;
              
                add_to_vectors(curr_point, norm, negative_OA_dot_norm, weight);

            }
        }

        valid_num += curr_point_list.size();
        for (int i = 0; i < (int)curr_point_list.size(); i++){
            ceres::CostFunction *cost_function = new SurfbackCostFunction(curr_point_list[i], norm_list[i], negative_OA_dot_norm_list[i], weight_list[i]);    
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
        if(curr_point_list.size()<20){
            ROS_WARN("not enough correct points %d", (int)curr_point_list.size());
            return Eigen::Isometry3d::Identity();
        }

        curr_point_list.clear();
        norm_list.clear();
        negative_OA_dot_norm_list.clear();
        weight_list.clear();
        //////////////////////////////////////////////////////////////////////////// back

        std::cout<<"t_add: "<<t_add.toc()<<"ms"<<std::endl;
        t_add_sum += t_add.toc();

        ROS_WARN("valid_num %d", valid_num);
        TicToc t_solve;

        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 1e-4;
        options.num_threads = std::thread::hardware_concurrency();
        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);
        std::cout<<"t_solve: "<<t_solve.toc()<<"ms"<<std::endl;
        t_solve_sum += t_solve.toc();

    }
    std::cout<<"before: "<<t_forward.x()<<" "<<t_forward.y()<<" "<<t_forward.z()<<std::endl;
    std::cout<<"after: "<<t_predict.x()<<" "<<t_predict.y()<<" "<<t_predict.z()<<std::endl;

    std::cout << "\033[32m"; // 设置文本颜色为绿色
    std::cout<<"t_add_sum: "<<t_add_sum<<std::endl;
    std::cout<<"t_solve_sum: "<<t_solve_sum<<std::endl;
    std::cout << "\033[0m";  // 重置文本颜色]"
    return qt2Isometry3d(q_predict, t_predict);
}


std::pair<Eigen::Isometry3d, double> loop_closure_process( pcl::PointCloud<PointType>::Ptr scan, 
                                Eigen::Isometry3d scan_pose,
                                pcl::PointCloud<PointType>::Ptr local_map,
                                Eigen::Isometry3d base_pose){
    TicToc t_loop;

    std::cout<< "\033[36m"; 
    Eigen::Isometry3d forward_pose =  base_pose.inverse() * scan_pose;
    Eigen::Quaterniond q_forward(forward_pose.rotation());
    Eigen::Vector3d t_forward = forward_pose.translation();
    double parameters[7] = {q_forward.x(), q_forward.y(), q_forward.z(), q_forward.w(), t_forward.x(), t_forward.y(), t_forward.z()};
    Eigen::Map<Eigen::Quaterniond> q_predict = Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_predict = Eigen::Map<Eigen::Vector3d>(parameters+4);

    std::shared_ptr<CELL> CELL_last_map = std::make_shared<CELL>(scan, index_tree, base_pose);
    CELL_last_map->from_ros_feature(local_map);

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    double inlier_thres_loop = 2;
    problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
    int valid_num = 0;
    for (int iter = 0; iter < 4; iter++){
        valid_num = 0;
        TicToc t_add;

        pcl::PointCloud<PointType>::Ptr scan_opti;
        pcl::PointCloud<PointType>::Ptr scan_base = scan;

    
        // if(iter != 0)
        scan_opti = TransformCloud(scan_base, q_predict, t_predict);
        // else
        // scan_opti = scan_base;

        CELL CELL_opti_current(scan_opti, index_tree, Eigen::Isometry3d::Identity());
        CELL_opti_current.calculate_feature_for(downsample_rate, false);
        // std::cout<<"feature time: "<<t_add.toc()<<"ms"<<std::endl;
        #pragma omp parallel for num_threads(16) 
        for (int i = 0; i < (int)scan_opti->points.size(); i++){
            ////// downsample
            if(i % downsample_rate!= 0)
                continue;
            //////

            int feature_id = CELL_opti_current.get_point_index(i);
            if((CELL_last_map->get_ballpoint_feature(feature_id) != 0))  
            //    (std::abs(CELL_local_map->get_ballpoint_feature(feature_id) - CELL_opti.get_ballpoint_feature(feature_id)) < high_threshs[iter < high_threshs.size() - 1? iter : high_threshs.size() - 1])
               {
                Eigen::Vector3d norm = CELL_last_map->get_ballpoint_normal(feature_id);
                if(norm.sum() == 0)
                    continue;
                
                // double negative_OA_dot_norm = 1 / norm.norm();
                double negative_OA_dot_norm = - CELL_last_map->get_ballpoint_feature(feature_id) *
                                                (index_tree->getInputCloud()->points[feature_id].x * norm.x() + 
                                                 index_tree->getInputCloud()->points[feature_id].y * norm.y() + 
                                                 index_tree->getInputCloud()->points[feature_id].z * norm.z());

         
                Eigen::Vector3d point_eigen(scan_opti->points[i].x, scan_opti->points[i].y, scan_opti->points[i].z);
                if (abs(norm.dot(point_eigen) + negative_OA_dot_norm) >  inlier_thres_loop) // / (iter + 1)
                    continue;
                CELL_last_map->set_regist_flag(feature_id);
                Eigen::Vector3d curr_point(scan_base->points[i].x, scan_base->points[i].y, scan_base->points[i].z);
                // double weight = abs(norm.dot(q_predict * CELL_current_map->get_ballpoint_normal(CELL_current_map->get_point_index(i))));
                // std::cout<<"forward i:"<<i<<" index:"<< CELL_current_map->get_point_index(i)<<" weight: "<<weight
                //         <<" normal:"<<CELL_current_map->get_ballpoint_normal(CELL_current_map->get_point_index(i))[0]<<std::endl;
                // if(weight < 0.95 )
                //     continue;
                add_to_vectors(curr_point, norm, negative_OA_dot_norm, 1);
            }
        }

        valid_num += curr_point_list.size();
        for (int i = 0; i < (int)curr_point_list.size(); i++){
            ceres::CostFunction *cost_function = new SurfForwardCostFunction(curr_point_list[i], norm_list[i], negative_OA_dot_norm_list[i], weight_list[i]);    
            problem.AddResidualBlock(cost_function, loss_function, parameters);
        }
        if(curr_point_list.size()<200){
            ROS_WARN("not enough correct points %d", (int)curr_point_list.size());
            return {Eigen::Isometry3d::Identity(), 0};
        }

        curr_point_list.clear();
        norm_list.clear();
        negative_OA_dot_norm_list.clear();
        weight_list.clear();
        // forward
       

        // std::cout<<"t_add: "<<t_add.toc()<<"ms"<<std::endl;
        // std::cout<<"valid_num"<< valid_num<<"/"<<(int)scan->points.size()*(iter+1)<<std::endl;

        
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 1e-4;
        options.num_threads = std::thread::hardware_concurrency();
        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);

    }
    // if(t_predict.norm() > 6){
    //     std::cout<< "\033[0m"; 
    //     return {Eigen::Isometry3d::Identity(), 0};
    // }
    std::cout<<"before: "<<t_forward.x()<<" "<<t_forward.y()<<" "<<t_forward.z()<<std::endl;
    std::cout<<"after: "<<t_predict.x()<<" "<<t_predict.y()<<" "<<t_predict.z()<<std::endl;
    std::cout<<"valid_num: "<<valid_num<<" / "<<(int)scan->points.size()<<std::endl;
    std::cout<< "\033[0m"; 

    std::cout << "\033[32m"; // 设置文本颜色为绿色
    std::cout<<"t_loop: "<<t_loop.toc()<<"ms"<<std::endl;
    std::cout << "\033[0m";  // 重置文本颜色]"
  
    return {qt2Isometry3d(q_predict, t_predict), (double)valid_num / (double)scan->points.size()};
}

gtsam::Pose3 eigen2gtsamPose(Eigen::Isometry3d pose){
    Eigen::Vector3d euler_angles = pose.linear().eulerAngles(2, 1, 0);

    return gtsam::Pose3(gtsam::Rot3::RzRyRx(euler_angles[2], euler_angles[1], euler_angles[0]), 
                        gtsam::Point3(pose.translation().x(), pose.translation().y(), pose.translation().z()));
}

Eigen::Isometry3d gtsamPose2eigen(gtsam::Pose3 pose)
{
    // Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    // matrix(0,3) = pose.translation().x();
    // matrix(1,3) = pose.translation().y();
    // matrix(2,3) = pose.translation().z();
    // matrix.block<3,3>(0,0) = pose.rotation().matrix().cast<double>();

    Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
    // Assign translation
    isometry.translation().x() = pose.translation().x();
    isometry.translation().y() = pose.translation().y();
    isometry.translation().z() = pose.translation().z();
    // Assign rotation
    isometry.linear() = pose.rotation().matrix().cast<double>();;
    return isometry;
}

void LiDARscanHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_lock.lock();
    LiDARscanBuf.push(msg);
    mutex_lock.unlock();
}



void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg)
{
    double timestamp = msg->header.stamp.toSec();
    pose_time_list.push_back(timestamp);
    Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();
    current_pose.rotate(Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z));  
    current_pose.pretranslate(Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));
    mtx_path.lock();
    pose_list.push_back(current_pose);
    mtx_path.unlock();

    if(pose_list.size() == 1){
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());
        // gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        mutex_gtsam.lock();
        initialEstimate.insert(0, eigen2gtsamPose(current_pose));
        prior_factor = gtsam::PriorFactor<gtsam::Pose3>(0, eigen2gtsamPose(current_pose), priorNoise);
        gtSAMgraph.add(prior_factor);
        // gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(0, 0, gtsam::Pose3(), priorNoise));
        mutex_gtsam.unlock();
        std::cout<<"prior factor"<<std::endl;
    }
    else{
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = eigen2gtsamPose(pose_list[pose_list.size()-2]);
        gtsam::Pose3 poseTo   = eigen2gtsamPose(current_pose);
        
        mutex_gtsam.lock();
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(pose_list.size()-2, pose_list.size()-1, poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(pose_list.size()-1, poseTo);
        mutex_gtsam.unlock();
    }
    publish_path(pose_list, pubbefore_path, timestamp);

}


std::vector<int> loop_detection(std::shared_ptr<CELL> CELL_current_map){ //return index in base_id_list
    PointType current_pose;
    current_pose.x = CELL_current_map->get_base_pose().translation().x(); 
    current_pose.y = CELL_current_map->get_base_pose().translation().y(); 
    current_pose.z = CELL_current_map->get_base_pose().translation().z();
    std::vector<int> loop_id_candidates;
    if(pose_tree->getInputCloud()->points.size() == 0)
        return loop_id_candidates;
    int loop_num = 3;
    std::vector<int> pointIdxNKNSearch(loop_num);
    std::vector<float> pointNKNSquaredDistance(loop_num);

    pose_tree->nearestKSearch(current_pose, loop_num, pointIdxNKNSearch, pointNKNSquaredDistance); // index in base_id_list
    std::cout<< "\033[36m"; 
    std::cout<<"fine loop closure between: "<<CELL_current_map->base_id<<" and "<<std::endl;
    for(size_t j = 0; j < loop_num; j ++){
        if(j == pose_tree->getInputCloud()->points.size())
            break;
        if(base_id_list.size() < loop_num || pointIdxNKNSearch[j] > (int)base_id_list.size() - loop_num -1 || 
            pointNKNSquaredDistance[j] > 144){
            std::cout<<base_id_list[pointIdxNKNSearch[j]]<<":"<<pointNKNSquaredDistance[j]<<std::endl;
            continue;
        }
        if(pointNKNSquaredDistance[j] < 144){
            loop_id_candidates.push_back(pointIdxNKNSearch[j]);  
            std::cout<<base_id_list[pointIdxNKNSearch[j]]<<":"<<
                 opti_pose_list[base_id_list[pointIdxNKNSearch[j]]].translation().x()<<" "<<
                 opti_pose_list[base_id_list[pointIdxNKNSearch[j]]].translation().y()<<" "<<
                 opti_pose_list[base_id_list[pointIdxNKNSearch[j]]].translation().z()<<" "<<
                 pointNKNSquaredDistance[j]<<" base index:"<<pointIdxNKNSearch[j]<<"/"<<(int)base_id_list.size() - loop_num -1 <<std::endl;
        }
    }
    std::cout << "\033[0m";  
    return loop_id_candidates;
}
void update_pose_to_file(const std::string& output_mode = "tum"){
    mtx_path.lock();
    isamCurrentEstimate = isam->calculateEstimate();
    opti_pose_list.clear();
    std::ofstream outfile;
    outfile.open(odom_path, ios::out | ios::trunc);
    
    for (int i = 0; i < (int)isamCurrentEstimate.size(); ++i) {
        Eigen::Matrix4d rm = gtsamPose2eigen(isamCurrentEstimate.at<gtsam::Pose3>(i)).matrix();
        
        if(output_mode == "kitti") {
            // KITTI format: 3x4 transformation matrix
            outfile << rm(0,0) << " " << rm(0,1) << " " << rm(0,2) << " " << rm(0,3) << " "
                   << rm(1,0) << " " << rm(1,1) << " " << rm(1,2) << " " << rm(1,3) << " "
                   << rm(2,0) << " " << rm(2,1) << " " << rm(2,2) << " " << rm(2,3) << std::endl;
        } 
        else if(output_mode == "tum") {
            // TUM format: timestamp x y z qx qy qz qw
            Eigen::Quaterniond q(rm.block<3,3>(0,0));
            outfile << std::fixed << std::setprecision(6) 
                   << pose_time_list[i] << " "
                   << rm(0,3) << " " << rm(1,3) << " " << rm(2,3) << " "
                   << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        
        opti_pose_list.push_back(gtsamPose2eigen(isamCurrentEstimate.at<gtsam::Pose3>(i)));
    }
    outfile.close();
    mtx_path.unlock();

}
void build_pose_tree(){
    pose_tree = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr pose_cloud(new pcl::PointCloud<PointType>());
    mtx_path.lock();
    if(base_id_list.size() == 0)
        return;
    for(size_t i = 0; i < base_id_list.size(); i++){
        PointType pt;
        pt.x = opti_pose_list[base_id_list[i]].translation().x();
        pt.y = opti_pose_list[base_id_list[i]].translation().y();
        pt.z = opti_pose_list[base_id_list[i]].translation().z();
        pose_cloud->points.push_back(pt);
    }
    mtx_path.unlock();
    pose_tree->setInputCloud(pose_cloud);
}
void LocalmapHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{   
    double timestamp = msg->header.stamp.toSec();
    int pose_i = 0;
    while(pose_time_list.empty() || timestamp > pose_time_list[pose_time_list.size()-1]){
        std::chrono::milliseconds dura(500);
        std::this_thread::sleep_for(dura); 
    }

    while(pose_i < (int)pose_time_list.size() && timestamp != pose_time_list[pose_i] )
        pose_i++;
    mutex_lock.lock();

    if(timestamp != pose_time_list[pose_i])
        ROS_WARN("localmap time not match with pose time");

    while(!LiDARscanBuf.empty() && LiDARscanBuf.front()->header.stamp.toSec() < timestamp)
        LiDARscanBuf.pop();
    pcl::PointCloud<PointType>::Ptr scan_raw(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*LiDARscanBuf.front(), *scan_raw);
    LiDARscanBuf.pop();
    mutex_lock.unlock();
    TicToc t_tfbetween;
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

    pcl::PointCloud<pcl::PointXYZI>::Ptr localmap(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*msg, *localmap);
    local_map_list.push_back(localmap); // for loop closure

    if(base_id_list.size() == 0){
        CELL_last_local_map = std::make_shared<CELL>(scan, index_tree, pose_list[pose_i]);
        CELL_last_local_map->calculate_feature_for(downsample_rate, false);
        CELL_last_local_map->from_ros_feature(localmap);
        base_id_list.push_back(0);
        CELL_last_local_map->base_id = 0;
    }
    else{
        base_id_list.push_back(pose_i);
        CELL_current_local_map = std::make_shared<CELL>(scan, index_tree, pose_list[pose_i]);
        CELL_current_local_map->calculate_feature_for(downsample_rate, false);
        CELL_current_local_map->from_ros_feature(localmap);
        CELL_current_local_map->base_id = pose_i;
        
        Eigen::Isometry3d optiTF = tfbetweenCELL(CELL_last_local_map, CELL_current_local_map); // bifirection

        gtsam::Pose3 poseBetween = eigen2gtsamPose(optiTF);
        double weight = 3;
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6 *weight, 1e-6 *weight, 1e-6 *weight, 1e-4 *weight, 1e-4 *weight, 1e-4 *weight).finished());

        mutex_gtsam.lock();
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(base_id_list[base_id_list.size()-2], base_id_list[base_id_list.size()-1], poseBetween, noiseBetween));
        std::cout<<"update isam"<<std::endl;
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        gtSAMgraph.resize(0);
        initialEstimate.clear();
        update_pose_to_file(output_mode);
        build_pose_tree();
        mutex_gtsam.unlock();

        std::cout<<"add base-base factor between:"<<base_id_list[base_id_list.size()-2] <<" and "<<base_id_list[base_id_list.size()-1]<<std::endl;
        ROS_WARN("t_tfbetween: %lf", t_tfbetween.toc());

        ///// loop closure
        if(FLAG_LOOP){
            TicToc t_loop_detection;
            // CELL_current_local_map->base_pose = opti_pose_list[opti_pose_list.size()-1];
            std::vector<int> loop_id_candidates = loop_detection(CELL_current_local_map);
            if(loop_id_candidates.size() > 0){
                for(size_t k = 0; k < 1; k++){ //loop_id_candidates.size()
                    std::pair<Eigen::Isometry3d, double> loop_out = loop_closure_process(scan, opti_pose_list[pose_i],
                            local_map_list[loop_id_candidates[k]], opti_pose_list[base_id_list[loop_id_candidates[k]]]);
                    Eigen::Isometry3d loopTF = loop_out.first;// calculate tf between loop
                    double valid_num = loop_out.second * downsample_rate;
                    std::cout<<"weight: "<<valid_num<<std::endl;
                    if(valid_num < 0.4)
                        continue;
                    gtsam::Pose3 poseLoop = eigen2gtsamPose(loopTF);
                    double weight = 1 / valid_num; // 1
                    gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6 *weight, 1e-6 *weight, 1e-6 *weight, 1e-4 *weight, 1e-4 *weight, 1e-4 *weight).finished());
                    mutex_gtsam.lock();
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(base_id_list[loop_id_candidates[0]], base_id_list[base_id_list.size()-1], poseLoop, noiseBetween));
                    mutex_gtsam.unlock();
                    pulish_loop_line(pose_list[base_id_list[loop_id_candidates[k]]],
                                 pose_list[base_id_list[base_id_list.size()-1]],
                                 publoop_line,
                                 timestamp, "o");
                    // pulish_loop_line(opti_pose_list[base_id_list[loop_id_candidates[k]]],
                    //                 opti_pose_list[base_id_list[base_id_list.size()-1]],
                    //                 publoop_line,
                    //                 timestamp, "b");
                }
                mutex_gtsam.lock();
                std::cout<<"update isam"<<std::endl;
                isam->update(gtSAMgraph, initialEstimate);
                isam->update();
                isam->update();
                isam->update();
                isam->update();
                gtSAMgraph.resize(0);
                initialEstimate.clear();
                update_pose_to_file(output_mode);
                build_pose_tree();
                mutex_gtsam.unlock();
                
                publish_path(opti_pose_list, pubafter_path);
            }
            //////loop closure
        }
        CELL_last_local_map = CELL_current_local_map; 
    }
}




void write_pose(const std::string& output_mode = "tum"){
    int write_flag = 0;
    while(1){
        mutex_gtsam.lock();
        if(gtSAMgraph.size() != 0){
            std::cout<<"update isam"<<std::endl;
            isam->update(gtSAMgraph, initialEstimate);
            isam->update();
            gtSAMgraph.resize(0);
            initialEstimate.clear();
            write_flag = 1;
        }
        if(opti_pose_list.size() != isam->calculateEstimate().size())
            write_flag = 1;
        mutex_gtsam.unlock(); 
        if(write_flag){
            update_pose_to_file(output_mode);
            write_flag = 0;
        }
        std::cout<<"after optimize, write_pose"<<std::endl;
        publish_path(opti_pose_list, pubafter_path);
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura); 
    }
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh("~"); 

    std::cout<<"main function"<<std::endl;

    int param_ball_num = 50000;

    std::string topic = "/rslidar_points";

    nh.getParam("param_ball_num", param_ball_num); 
    nh.getParam("param_iter", param_iter);
    nh.getParam("downsample_rate", downsample_rate);
    nh.getParam("inlier_thres", inlier_thres);
    nh.getParam("ball_height", ball_height);
    nh.getParam("FLAG_LOOP", FLAG_LOOP);

    nh.getParam("odom_path", odom_path);
    nh.getParam("output_mode", output_mode);
    nh.getParam("cell_odom_path", cell_odom_path);

    outfile.open(odom_path, ios::out | ios::trunc);
    cell_outfile.open(cell_odom_path, ios::out | ios::trunc);
    
    nh.getParam("topic", topic);
    std::cout<<"input topic:"<< topic<<std::endl;

    index_tree =  pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    index_tree->setInputCloud(generate_ball(param_ball_num));


    ros::Subscriber subLiDARscan = nh.subscribe<sensor_msgs::PointCloud2>(topic, 1000, LiDARscanHandler);
    ros::Subscriber subodom = nh.subscribe<nav_msgs::Odometry>("/odom", 1000, OdomHandler);
    ros::Subscriber subLocalmap = nh.subscribe<sensor_msgs::PointCloud2>("/ros_feature", 1000, LocalmapHandler);

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    // pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    // pubball_scan = nh.advertise<sensor_msgs::PointCloud2>("/ball_scan", 1000);
    // pubtfscan = nh.advertise<sensor_msgs::PointCloud2>("/tfscan", 1000);
    // pubball_local_map = nh.advertise<sensor_msgs::PointCloud2>("/ball_local_map", 1000);
    // pubball_global_map = nh.advertise<sensor_msgs::PointCloud2>("/ball_global_map", 1000);
    // pubpointcloud_feature = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_feature", 1000); 
    // pubpointcloud_indice = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_indice", 1000); 
    // pubpointcloud_normal = nh.advertise<sensor_msgs::PointCloud2>("/pointcloud_normal", 1000); 
    // pubros_feature = nh.advertise<sensor_msgs::PointCloud2>("/ros_feature", 1000);
    publoop_CELL = nh.advertise<sensor_msgs::PointCloud2>("/loop_feature", 1000); 
    publoop_scan = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan", 1000); ;
    pubbefore_path = nh.advertise<nav_msgs::Path>("/before_path", 1000);
    pubafter_path = nh.advertise<nav_msgs::Path>("/after_path", 1000);
    publoop_line = nh.advertise<visualization_msgs::Marker>("/loop_detection", 10);

    std::thread write_pose_process{write_pose, output_mode};

    ros::MultiThreadedSpinner spinner(4); // 使用4个线程
    spinner.spin();
    // ros::spin();

    return 0;
}

