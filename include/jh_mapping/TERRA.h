#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>  
#include <pcl/visualization/pcl_visualizer.h> 

#include <ros/ros.h>
#include <CELLmap/common.h>
#include <CELLmap/tic_toc.h>

class TERRA {
public:

    double new_num;
    Eigen::VectorXd degenaracy_value;
    Eigen::Matrix3d degenaracy_direction;
    int base_id;
    Eigen::Isometry3d base_pose;

    TERRA() {}
    TERRA(const pcl::PointCloud<PointType>::Ptr& cloud, const pcl::KdTreeFLANN<PointType>::Ptr& index_tree, const Eigen::Isometry3d& base_pose, const double timestamp=0)
        : cloud(cloud), index_tree(index_tree), base_pose(base_pose), timestamp(timestamp),
          feature_length(index_tree->getInputCloud()->points.size()), 
          feature(Eigen::VectorXd::Zero(feature_length)), 
          ball_distence(Eigen::VectorXd::Zero(feature_length)), // 球面距离
          normals(feature_length, Eigen::Vector3d::Zero()), 
          area_points(feature_length), 
          indices(Eigen::VectorXd::Zero(cloud->points.size())),
          centers(feature_length, Eigen::Vector3d::Zero()),
          area_point_num(feature_length, 0),
          covMat(feature_length, Eigen::Matrix3d::Zero()),
          regist_flag(Eigen::VectorXd::Zero(feature_length)),
          min_eigenvalue(Eigen::VectorXd::Zero(feature_length)),
          new_num(0),
          degenaracy_value(Eigen::Vector3d::Zero()),
          degenaracy_direction(Eigen::Matrix3d::Zero()){
        if (!cloud || !index_tree) {
            throw std::invalid_argument("Invalid input: cloud or index_tree is null");
        }
        initializeAreaPoints();
    }
    ~TERRA() {
        // 释放资源
    }

    void initializeAreaPoints() {  
        for (size_t i = 0; i < feature_length; ++i) {  
            area_points[i] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);  
        }  

    }  
    void calculate_feature_for(int downsample_rate, bool FLAG_tf = true) {
        pcl::PointCloud<PointType>::Ptr tfcloud;
        if(FLAG_tf)
            tfcloud = TransformCloud(cloud, base_pose.inverse());
        else
            tfcloud = cloud;

        #pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < tfcloud->points.size(); ++i) {
            if(i % downsample_rate!= 0)
                continue;
            Eigen::Vector3d point(tfcloud->points[i].x, tfcloud->points[i].y, tfcloud->points[i].z);
            double distance = point.norm();
            if(distance == 0)
                continue;

            Eigen::Vector3d point_normalized(point(0) / distance, point(1) / distance, point(2) / distance);
            int k = 1;  
            std::vector<int> pointIdxNKNSearch(k);
            std::vector<float> pointNKNSquaredDistance(k);
            PointType point_normalized_pcl;
            point_normalized_pcl.x = point_normalized(0); 
            point_normalized_pcl.y = point_normalized(1);
            point_normalized_pcl.z = point_normalized(2);
        
            index_tree->nearestKSearch(point_normalized_pcl, k, pointIdxNKNSearch, pointNKNSquaredDistance);
            for(int j = 0; j < k; j ++){
                if(feature[pointIdxNKNSearch[j]] == 0){
                    // feature[pointIdxNKNSearch[j]] = distance;
                    ball_distence[pointIdxNKNSearch[j]] = pointNKNSquaredDistance[j];
                }
                else{
                    // #pragma omp critical
                    if(pointNKNSquaredDistance[j] < ball_distence[pointIdxNKNSearch[j]]){
                        ball_distence[pointIdxNKNSearch[j]] = pointNKNSquaredDistance[j];
                        // feature[pointIdxNKNSearch[j]] = distance;
                    }
                }
                indices[i] = pointIdxNKNSearch[j];
                #pragma omp critical
                {
                // if (!area_points[indices[i]]) {
                //     area_points[indices[i]] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
                // }
                area_points[indices[i]]->push_back(tfcloud->points[i]);
                }

            }
        } 
    }




    void calculate_normal(bool FLAG_tf = false) {
        pcl::PointCloud<PointType>::Ptr tfcloud;
        if(FLAG_tf)
            tfcloud = TransformCloud(cloud, base_pose.inverse());
        else
            tfcloud = cloud;
        // pcl::PointCloud<PointType>::Ptr tfcloud = TransformCloud(cloud, base_pose.inverse());

        for (size_t i = 0; i < tfcloud->points.size(); ++i) {
            if (!area_points[indices[i]]) {
                area_points[indices[i]] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
            }
            area_points[indices[i]]->push_back(tfcloud->points[i]);
        }
        int normal_num = 0;
        int dayu5_num = 0;
        for (int i = 0; i < feature_length; i++) {
            if(!area_points[i])
                continue;
            int point_size = area_points[i]->points.size();
            if (point_size > 5) {
                dayu5_num++;

                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < point_size; j++)
                {
                    Eigen::Vector3d tmp(area_points[i]->points[j].x,
                                        area_points[i]->points[j].y,
                                        area_points[i]->points[j].z);
                    center = center + tmp;
                    nearCorners.push_back(tmp);
                }
                center = center / point_size;

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for (int j = 0; j < point_size; j++){
                    Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                    covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                if (saes.eigenvalues()[1] > 5 * saes.eigenvalues()[0]){
                    Eigen::Vector3d normal = saes.eigenvectors().col(0);
                    // if (normal.dot(index_tree[i]) >= 0) {
                    //     normal.row(i) = -normal.transpose();
                    // } else {
                    //     normal.row(i) = normal.transpose();
                    // }
                    normal_num++;
                    normals[i] = normal;
                }

            }
        }
        std::cout<< "normal_num: " << normal_num << std::endl;
    }


    void calculate_normal_and_feature(int plane_thres, int area_point_thres, bool FLAG_tf = true) {
        TicToc t_normal_feature;
        pcl::PointCloud<PointType>::Ptr tfcloud;
        if(FLAG_tf)
            tfcloud = TransformCloud(cloud, base_pose.inverse());
        else
            tfcloud = cloud;

        std::vector< std::vector<Eigen::Vector3d>> nearCorners(feature_length, std::vector<Eigen::Vector3d>());
        int valid_num = 0;
        #pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < tfcloud->points.size(); ++i) {

            Eigen::Vector3d point(tfcloud->points[i].x, tfcloud->points[i].y, tfcloud->points[i].z);
            double distance = point.norm();
            if(distance == 0)
                continue;
            Eigen::Vector3d point_normalized(point(0) / distance, point(1) / distance, point(2) / distance);
            int k = 1;  
            std::vector<int> pointIdxNKNSearch(k);
            std::vector<float> pointNKNSquaredDistance(k);
            PointType point_normalized_pcl;
            point_normalized_pcl.x = point_normalized(0); 
            point_normalized_pcl.y = point_normalized(1);
            point_normalized_pcl.z = point_normalized(2);
        
            index_tree->nearestKSearch(point_normalized_pcl, k, pointIdxNKNSearch, pointNKNSquaredDistance);
            for(int j = 0; j < k; j ++){
                #pragma omp critical
                {
                nearCorners[pointIdxNKNSearch[j]].push_back(point);
                centers[pointIdxNKNSearch[j]] += point;
                }
                indices[i] = pointIdxNKNSearch[j];
            }
        } 
        std::cout<<"nearCorners and centers are calculated!"<<std::endl;

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < feature_length; i++) {
            if(nearCorners[i].size() == 0)
                continue;
            area_point_num[i] = nearCorners[i].size();
            if (area_point_num[i] > area_point_thres) {
                centers[i] = centers[i] / area_point_num[i];

            
                /////////////////
                // for (int j = 0; j < point_size; j++){
                //     Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[i][j] - center;
                //     covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                // }
                /////////////////
                for (int j = 0; j < area_point_num[i]; j++){
                    // Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[i][j] - center;
                    covMat[i] = covMat[i] + nearCorners[i][j] * nearCorners[i][j].transpose();
                }
                covMat[i] = covMat[i] - area_point_num[i] * centers[i] * centers[i].transpose();
                ///////////////

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat[i]);
                if (saes.eigenvalues()[1] > plane_thres * saes.eigenvalues()[0]){ // &&  saes.eigenvalues()[0] < 0.01
                    Eigen::Vector3d normal = saes.eigenvectors().col(0);
                    // if (normal.dot(index_tree[i]) >= 0) {
                    //     normal.row(i) = -normal.transpose();
                    // } else {
                    //     normal.row(i) = normal.transpose();
                    // }
                    normals[i] = normal;
                    min_eigenvalue[i] = saes.eigenvalues()[0];
                    double D = - normals[i].dot(centers[i]);
                    Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
                                            index_tree->getInputCloud()->points[i].y, 
                                            index_tree->getInputCloud()->points[i].z);
                    double dis = - D / normals[i].dot(direct);
                    if(dis > 0 && dis < 100){
                        feature[i] = dis;
                        #pragma omp critical
                        valid_num += area_point_num[i];
                    }
                }
                // else{ //// use ransac for more plane
                //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);  
                //     std::cout<<"ransac:"<<area_point_num[i]<<std::endl;
                //     // 分配足够的空间给点云  
                //     cloud->width = area_point_num[i];  
                //     cloud->height = 1; 
                //     cloud->is_dense = true;  
                //     cloud->points.resize(cloud->width * cloud->height);  

                //     // 将 std::vector<Eigen::Vector3d> 转换为 pcl::PointCloud<pcl::PointXYZ>  
                //     for (size_t j = 0; j < area_point_num[i]; ++j) {  
                //         cloud->points[j].x = nearCorners[i][j](0);  
                //         cloud->points[j].y = nearCorners[i][j](1);  
                //         cloud->points[j].z = nearCorners[i][j](2);  
                //     }  

                //     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                //     //inliers表示误差能容忍的点 记录的是点云的序号
                //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                //     //创建分割器
                //     pcl::SACSegmentation<pcl::PointXYZ> seg;
                //     seg.setOptimizeCoefficients(true);
                //     seg.setModelType(pcl::SACMODEL_PLANE);
                //     seg.setMethodType(pcl::SAC_RANSAC);
                //     seg.setDistanceThreshold(0.03);
                //     seg.setMaxIterations(500);
                //     seg.setInputCloud(cloud);
                //     seg.segment(*inliers, *coefficients);
              
                //     if(inliers->indices.size() > area_point_thres){

                //         std::cout<<"inlier num:"<< inliers->indices.size()<<" / "<< area_point_num[i]<<std::endl;
                //         std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                //             << coefficients->values[1] << " "
                //             << coefficients->values[2] << " "
                //             << coefficients->values[3] << std::endl;


                //         centers[i] = Eigen::Vector3d::Zero();
                //         covMat[i] = Eigen::Matrix3d::Zero();
                //         area_point_num[i] = inliers->indices.size();

                //         for(int j =0; j < inliers->indices.size(); j++){
                //             Eigen::Vector3d tmp(cloud->points[inliers->indices[j]].x,
                //                             cloud->points[inliers->indices[j]].y,
                //                             cloud->points[inliers->indices[j]].z);
                //             centers[i] = centers[i] + tmp;
                //             covMat[i] = covMat[i] + tmp * tmp.transpose();

                //         }
                //         centers[i] = centers[i] / area_point_num[i];
                //         covMat[i] = covMat[i] - area_point_num[i] * centers[i] * centers[i].transpose();
                //         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat[i]);
                //         if (saes.eigenvalues()[1] > plane_thres * saes.eigenvalues()[0] && saes.eigenvalues()[0] < 0.01){
                //             Eigen::Vector3d normal = saes.eigenvectors().col(0);
                //             normals[i] = normal;
                //             min_eigenvalue[i] = saes.eigenvalues()[0];
                //             double D = - normals[i].dot(centers[i]);
                //             Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
                //                                     index_tree->getInputCloud()->points[i].y, 
                //                                     index_tree->getInputCloud()->points[i].z);
                //             double dis = - D / normals[i].dot(direct);
                //             if(dis > 0 && dis < 100)
                //                 feature[i] = dis;
                //         }
                //     }
                // }

            }
        }
        std::cout<<"calculate_normal_and_feature svalid_num: "<<valid_num <<" / "<<tfcloud->points.size()<<std::endl;
        std::cout<<"t_normal_feature: "<<t_normal_feature.toc()<<std::endl;
        
        // ROS_WARN("calculate_normal_and_feature time: %f", t.toc());
    }

    void calculate_normal_and_feature2(int plane_thres, int area_point_thres, bool FLAG_tf = true) {
        TicToc t_normal_feature;
        pcl::PointCloud<PointType>::Ptr tfcloud;
        if(FLAG_tf)
            tfcloud = TransformCloud(cloud, base_pose.inverse());
        else
            tfcloud = cloud;

        std::vector< std::vector<Eigen::Vector3d>> nearCorners(feature_length, std::vector<Eigen::Vector3d>());

        #pragma omp parallel for num_threads(16)
        for (size_t i = 0; i < tfcloud->points.size(); ++i) {

            Eigen::Vector3d point(tfcloud->points[i].x, tfcloud->points[i].y, tfcloud->points[i].z);
            double distance = point.norm();
            if(distance == 0)
                continue;
            Eigen::Vector3d point_normalized(point(0) / distance, point(1) / distance, point(2) / distance);
            int k = 1;  
            std::vector<int> pointIdxNKNSearch(k);
            std::vector<float> pointNKNSquaredDistance(k);
            PointType point_normalized_pcl;
            point_normalized_pcl.x = point_normalized(0); 
            point_normalized_pcl.y = point_normalized(1);
            point_normalized_pcl.z = point_normalized(2);
        
            index_tree->nearestKSearch(point_normalized_pcl, k, pointIdxNKNSearch, pointNKNSquaredDistance);
            for(int j = 0; j < k; j ++){
                #pragma omp critical
                {
                nearCorners[pointIdxNKNSearch[j]].push_back(point);
                centers[pointIdxNKNSearch[j]] += point;
                }
                indices[i] = pointIdxNKNSearch[j];
            }
        } 
        std::cout<<"nearCorners and centers are calculated!"<<std::endl;

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < feature_length; i++) {
            if ((int)nearCorners[i].size() < area_point_thres) 
                continue;
            area_point_num[i] = nearCorners[i].size();

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);  
            // std::cout<<"ransac:"<<area_point_num[i]<<std::endl;
            // 分配足够的空间给点云  
            cloud->width = area_point_num[i];  
            cloud->height = 1; 
            cloud->is_dense = true;  
            cloud->points.resize(cloud->width * cloud->height);  

            // 将 std::vector<Eigen::Vector3d> 转换为 pcl::PointCloud<pcl::PointXYZ>  
            for (size_t j = 0; j < area_point_num[i]; ++j) {  
                cloud->points[j].x = nearCorners[i][j](0);  
                cloud->points[j].y = nearCorners[i][j](1);  
                cloud->points[j].z = nearCorners[i][j](2);  
            }  

            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            //inliers表示误差能容忍的点 记录的是点云的序号
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            //创建分割器
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.03);
            seg.setMaxIterations(500);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);
        
            if((int)inliers->indices.size() > area_point_thres){

                // std::cout<<"inlier num:"<< inliers->indices.size()<<" / "<< area_point_num[i]<<std::endl;
                // std::cout << "Model coefficients: " << coefficients->values[0] << " "
                //     << coefficients->values[1] << " "
                //     << coefficients->values[2] << " "
                //     << coefficients->values[3] << std::endl;


                centers[i] = Eigen::Vector3d::Zero();
                covMat[i] = Eigen::Matrix3d::Zero();
                area_point_num[i] = inliers->indices.size();

                for(int j =0; j < (int) inliers->indices.size(); j++){
                    Eigen::Vector3d tmp(cloud->points[inliers->indices[j]].x,
                                    cloud->points[inliers->indices[j]].y,
                                    cloud->points[inliers->indices[j]].z);
                    centers[i] = centers[i] + tmp;
                    covMat[i] = covMat[i] + tmp * tmp.transpose();

                }
                centers[i] = centers[i] / area_point_num[i];
                covMat[i] = covMat[i] - area_point_num[i] * centers[i] * centers[i].transpose();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat[i]);
                if (saes.eigenvalues()[1] > plane_thres * saes.eigenvalues()[0]){// && saes.eigenvalues()[0] < 0.01
                    Eigen::Vector3d normal = saes.eigenvectors().col(0);
                    normals[i] = normal;
                    min_eigenvalue[i] = saes.eigenvalues()[0];
                    double D = - normals[i].dot(centers[i]);
                    Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
                                            index_tree->getInputCloud()->points[i].y, 
                                            index_tree->getInputCloud()->points[i].z);
                    double dis = - D / normals[i].dot(direct);
                    if(dis > 0 && dis < 100)
                        feature[i] = dis;
                }
            }
        }
        std::cout<<"t_normal_feature: "<<t_normal_feature.toc()<<std::endl;
    }


    template <typename T>  
    void printVector(const std::vector<T>& vec) {  
        for (const auto& elem : vec) {  
            std::cout << elem << " ";  
        }  
        std::cout << std::endl;  
    }

    double the_biggest_distance_gap(std::vector<double> &distance_list){

        std::vector<size_t> indices(distance_list.size());  
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., n-1  
        std::sort(indices.begin(), indices.end(),   
                [&](size_t i1,size_t i2) { return distance_list[i1] < distance_list[i2]; });  

        std::vector<double> sorted_array(distance_list.size());  
        for (size_t i = 0; i < indices.size(); ++i) {  
            sorted_array[i] = distance_list[indices[i]];  
        }  
        
        // std::cout << "Sorted array: ";  
        // printVector(sorted_array);  

        // Step 2: Compute consecutive differences  
        std::vector<double> differences(sorted_array.size() - 1);
        for(size_t i = 0; i < differences.size() - 1; ++i) {
            differences[i] = sorted_array[i+1] - sorted_array[i];  
        }
        
        // std::cout << "Differences: ";  
        // printVector(differences);  
        auto max_it = std::max_element(differences.begin(), differences.end());  
        double max_value = *max_it;  

        return max_value;  
    }

    void calculate_normal_and_feature3(int plane_thres, int area_point_thres, double ransac_thres, bool FLAG_tf = true) {
        TicToc t_normal_feature;

  
        ///////////////////////for visual
          
        // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));  
        // viewer->setBackgroundColor(0, 0, 0); // 设置背景颜色（黑色）  
        ///////////////////////for visual
              


        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < feature_length; i++) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            if ((int)area_points[i]->points.size() < area_point_thres) 
                continue;
            area_point_num[i] = area_points[i]->points.size();

            ///////////////////////for visual
            // for(size_t j = 0; j < area_points[i]->points.size(); j++){
            //     pcl::PointXYZRGB pointRGB;  

            //     pointRGB.x = area_points[i]->points[j].x;  
            //     pointRGB.y = area_points[i]->points[j].y;  
            //     pointRGB.z = area_points[i]->points[j].z;  

            //     pointRGB.r = 255;  
            //     pointRGB.g = 255;  
            //     pointRGB.b = 255;  
            //     cloud->points.push_back(pointRGB);
            // }
            ///////////////////////for visual


            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            //inliers表示误差能容忍的点 记录的是点云的序号
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            //创建分割器
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(ransac_thres);
            seg.setMaxIterations(500);
            seg.setInputCloud(area_points[i]);
            seg.segment(*inliers, *coefficients);
        
            if((int)inliers->indices.size() > area_point_thres){

                // std::cout<<"inlier num:"<< inliers->indices.size()<<" / "<< area_point_num[i]<<std::endl;
                // std::cout << "Model coefficients: " << coefficients->values[0] << " "
                //     << coefficients->values[1] << " "
                //     << coefficients->values[2] << " "
                //     << coefficients->values[3] << std::endl;


                centers[i] = Eigen::Vector3d::Zero();
                covMat[i] = Eigen::Matrix3d::Zero();
                area_point_num[i] = inliers->indices.size();
                std::vector<double> distance_list;
                for(int j =0; j < (int) inliers->indices.size(); j++){
            ///////////////////////for visual

                    // pcl::PointXYZRGB point;
                    // point.x = cloud->points[inliers->indices[j]].x;
                    // point.y = cloud->points[inliers->indices[j]].y;
                    // point.z = cloud->points[inliers->indices[j]].z;
                    // point.r = 255;
                    // point.g = 0;
                    // point.b = 0;
                    // inlier_cloud->points.push_back(point);
            ///////////////////////for visual

                    Eigen::Vector3d tmp(area_points[i]->points[inliers->indices[j]].x,
                                    area_points[i]->points[inliers->indices[j]].y,
                                    area_points[i]->points[inliers->indices[j]].z);
                    distance_list.push_back(tmp.norm());
                    centers[i] = centers[i] + tmp;
                    covMat[i] = covMat[i] + tmp * tmp.transpose();

                }
                // if(the_biggest_distance_gap(distance_list) > 3)
                //     continue;
                centers[i] = centers[i] / area_point_num[i];
                covMat[i] = covMat[i] - area_point_num[i] * centers[i] * centers[i].transpose();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat[i]);
                if (saes.eigenvalues()[1] > plane_thres * saes.eigenvalues()[0]){// && saes.eigenvalues()[0] < 0.01
                    Eigen::Vector3d normal = saes.eigenvectors().col(0);
                    normals[i] = normal;
                    min_eigenvalue[i] = saes.eigenvalues()[0];
                    double D = - normals[i].dot(centers[i]);
                    Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
                                            index_tree->getInputCloud()->points[i].y, 
                                            index_tree->getInputCloud()->points[i].z);
                    double dis = - D / normals[i].dot(direct);
                    if(dis > 0 && dis < 100){
                        feature[i] = dis;

                    }
                }
            }
            ///////////////////////for visual
            // if(i %  1000 == 0 && (int)inliers->indices.size() > area_point_thres){
            //     if(inlier_cloud->points.size() > 0){
            //         inlier_cloud->width = inlier_cloud->points.size();  
            //         inlier_cloud->height = 1; 
            //         inlier_cloud->is_dense = true; 
            //         pcl::io::savePCDFileASCII("/home/yjsx/tmp/inlier"+std::to_string(i)+".pcd", *inlier_cloud); 

            //     }
            //     if(cloud->points.size() > 0){
            //         cloud->width = cloud->points.size();  
            //         cloud->height = 1; 
            //         cloud->is_dense = true;  
            //         pcl::io::savePCDFileASCII("/home/yjsx/tmp/all"+std::to_string(i) +".pcd", *cloud); 
                    
            //     }
            // }
            // viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud"); // 添加点云  


            // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "sample cloud"); // 设置点云大小  
            // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 30, "sample cloud2"); // 设置点云大小  

            
            // viewer->addCoordinateSystem(1.0); // 添加坐标系  
            // viewer->initCameraParameters(); // 初始化相机参数  

            // // 进入主循环，显示点云  
            // while (!viewer->wasStopped())  
            // {  
            //     viewer->spinOnce(100);  
            //     std::this_thread::sleep_for(std::chrono::milliseconds(100));  
            // }
            ///////////////////////for visual
        }
        std::cout << "\033[32m"; // 设置文本颜色为绿色
        std::cout<<"t_normal_feature: "<<t_normal_feature.toc()<<std::endl;
        std::cout << "\033[0m";  // 重置文本颜色
    }

            
  
        // ROS_WARN("calculate_normal_and_feature time: %f", t.toc());
    

    // void update(pcl::PointCloud<PointType>::Ptr tfcloud, int plane_thres) {
    //     std::vector< std::vector<Eigen::Vector3d>> nearCorners(feature_length, std::vector<Eigen::Vector3d>());
    //     std::vector<Eigen::Vector3d> new_centers(feature_length, Eigen::Vector3d::Zero());
    //     // #pragma omp parallel for num_threads(16)
    //     for (size_t i = 0; i < tfcloud->points.size(); ++i) {
    //         Eigen::Vector3d point(tfcloud->points[i].x, tfcloud->points[i].y, tfcloud->points[i].z);
    //         double distance = point.norm();
    //         if(distance == 0)
    //             continue;
    //         Eigen::Vector3d point_normalized(point(0) / distance, point(1) / distance, point(2) / distance);
    //         int k = 1;  
    //         std::vector<int> pointIdxNKNSearch(k);
    //         std::vector<float> pointNKNSquaredDistance(k);
    //         PointType point_normalized_pcl;
    //         point_normalized_pcl.x = point_normalized(0); 
    //         point_normalized_pcl.y = point_normalized(1);
    //         point_normalized_pcl.z = point_normalized(2);
        
    //         index_tree->nearestKSearch(point_normalized_pcl, k, pointIdxNKNSearch, pointNKNSquaredDistance);
    //         for(int j = 0; j < k; j ++){
    //             if((feature[pointIdxNKNSearch[j]] != 0 && std::abs(normals[pointIdxNKNSearch[j]].dot(point) - normals[pointIdxNKNSearch[j]].dot(centers[pointIdxNKNSearch[j]])) <0.1) || feature[pointIdxNKNSearch[j]] == 0){
    //                 // std::cout<<std::abs(normals[pointIdxNKNSearch[j]].dot(point) - normals[pointIdxNKNSearch[j]].dot(centers[pointIdxNKNSearch[j]]))<<std::endl;
    //                 nearCorners[pointIdxNKNSearch[j]].push_back(point);
    //                 new_centers[pointIdxNKNSearch[j]] += point;
    //             }
    //             nearCorners[pointIdxNKNSearch[j]].push_back(point);
    //             new_centers[pointIdxNKNSearch[j]] += point;
    //             // indices[i] = pointIdxNKNSearch[j];
    //         }
    //     } 

    //     int update_num = 0;
    //     #pragma omp parallel for num_threads(16)

    //     for (int i = 0; i < feature_length; i++) {
    //         if(nearCorners[i].size() == 0)
    //             continue;
    //         int new_area_point_num = nearCorners[i].size();
    //         Eigen::Matrix3d new_covMat = Eigen::Matrix3d::Zero();
    //         Eigen::Vector3d all_center(0, 0, 0);
            
    //         if (new_area_point_num > 10) {
    //             new_covMat = new_covMat + covMat[i] + area_point_num[i] * centers[i] * centers[i].transpose();
                
    //             // covMat[i] = covMat[i] + area_point_num[i] * centers[i] * centers[i].transpose(); //pp

    //             all_center = (centers[i] * area_point_num[i] + new_centers[i]) / (area_point_num[i] + new_area_point_num);
    //             // area_point_num[i] = area_point_num[i] + new_area_point_num;

            
    //             /////////////////
    //             // for (int j = 0; j < point_size; j++){
    //             //     Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[i][j] - center;
    //             //     covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
    //             // }
    //             /////////////////
    //             for (int j = 0; j < new_area_point_num; j++){
    //                 // Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[i][j] - center;
    //                 new_covMat = new_covMat + nearCorners[i][j] * nearCorners[i][j].transpose();
    //             }
    //             new_covMat = new_covMat - (new_area_point_num + area_point_num[i]) * all_center * all_center.transpose();
    //             // covMat[i] = covMat[i] - area_point_num[i] * centers[i] * centers[i].transpose();
    //             ///////////////

    //             Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(new_covMat);

    //             if (saes.eigenvalues()[1] > 20 * saes.eigenvalues()[0]){
    //                 Eigen::Vector3d normal = saes.eigenvectors().col(0);

    //                 double D = - normal.dot(all_center);
    //                 Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
    //                                         index_tree->getInputCloud()->points[i].y, 
    //                                         index_tree->getInputCloud()->points[i].z);
    //                 double dis = - D / normal.dot(direct);
    //                 if(dis > 0 && dis < 100 && dis - feature[i] > 0.1){
    //                     // std::cout<<"update num: "<<new_area_point_num << " dis change: " << dis - feature[i] 
    //                     //          <<" center change: "<<(centers[i] - all_center).norm()
    //                     //          <<" normal change: "<<(normals[i] - normal).norm()<<std::endl;
                        
    //                     // if(dis - feature[i] > 1 || (normals[i] - normal).norm() > 1){
    //                     //     std::cout<<"old dis: "<<feature[i]<<" new dis: "<<dis<<std::endl;
    //                     //     std::cout<<"old center: "<<centers[i].transpose()<<" new center: "<<all_center.transpose()<<std::endl;
    //                     //     std::cout<<"old normal: "<<normals[i].transpose()<<" new normal: "<<normal.transpose()<<std::endl;
    //                     // }
    //                     feature[i] = dis;
    //                     normals[i] = normal;
    //                     covMat[i] = new_covMat;
    //                     area_point_num[i] = area_point_num[i] + new_area_point_num;
    //                     centers[i] = all_center;
    //                     update_num ++;

    //                 }
    //             }
    //         }
    //     } 
    //     std::cout<<"update num: "<<update_num<<std::endl;
    // }
    
    int update2(std::shared_ptr<TERRA> TERRA_frame, int plane_thres) {
        TicToc t_update;
        int update_num = 0;
        int occluded_num = 0;
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < feature_length; i++) {
            if(TERRA_frame->feature[i] == 0)// 未构成有效平面
                continue;
            if(feature[i] != 0){ //原来有平面
                // ROS_WARN("normal distance: %f, feature distance: %f", normals[i].dot(TERRA_frame->normals[i]), abs(feature[i] - TERRA_frame->feature[i]));
                if(abs(TERRA_frame->normals[i].dot(normals[i])) > 0.95 )// 更新不大
                    continue;
                if(abs(TERRA_frame->feature[i] - feature[i]) > 2){// 被遮挡的被看见了
                    #pragma omp atomic
                    occluded_num ++;
                    continue;
                }

                else{ // 更新
                    int new_area_point_num = area_point_num[i] + TERRA_frame->area_point_num[i];
                    Eigen::Vector3d new_center = (centers[i] * area_point_num[i] + TERRA_frame->centers[i] * TERRA_frame->area_point_num[i]) / (area_point_num[i] + TERRA_frame->area_point_num[i]);
                    Eigen::Matrix3d new_covMat = (covMat[i] + 
                                                area_point_num[i] * centers[i] * centers[i].transpose() + 
                                                TERRA_frame->covMat[i] + 
                                                TERRA_frame->area_point_num[i] * TERRA_frame->centers[i] * TERRA_frame->centers[i].transpose()) - 
                                                 new_area_point_num * new_center * new_center.transpose();

                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(new_covMat);

                    if (saes.eigenvalues()[1] > plane_thres * saes.eigenvalues()[0] ){ //&&  saes.eigenvalues()[0] < 0.01
                        Eigen::Vector3d normal = saes.eigenvectors().col(0);

                        double D = - normal.dot(new_center);
                        Eigen::Vector3d direct(index_tree->getInputCloud()->points[i].x, 
                                                index_tree->getInputCloud()->points[i].y, 
                                                index_tree->getInputCloud()->points[i].z);
                        double dis = - D / normal.dot(direct);
                        if(dis > 0 && dis < 100 && dis - feature[i] > 0.1){
                            min_eigenvalue[i] = saes.eigenvalues()[0];
                            feature[i] = dis;
                            normals[i] = normal;
                            covMat[i] = new_covMat;
                            area_point_num[i] = new_area_point_num;
                            centers[i] = new_center;
                            #pragma omp atomic
                            update_num ++;

                        }
                    }
                }
            }
            else{ //新增
                if(TERRA_frame->feature[i] != 0){
                    min_eigenvalue[i]   = TERRA_frame->min_eigenvalue[i];

                    feature[i]          = TERRA_frame->feature[i];
                    normals[i]          = TERRA_frame->normals[i];       
                    covMat[i]           = TERRA_frame->covMat[i];        
                    area_point_num[i]   = TERRA_frame->area_point_num[i];
                    centers[i]          = TERRA_frame->centers[i];    
                    #pragma omp atomic   
                    new_num ++;
                }
            }
        }
        
        std::cout<<"t_update: "<<t_update.toc()<<std::endl;
        std::cout << "\033[32m"; // 设置文本颜色为绿色
        std::cout << "update num: " << update_num << " new num: " << new_num << " occluded num: " << occluded_num << std::endl;
        std::cout << "\033[0m";  // 重置文本颜色
        return occluded_num;
    }

    int update3(std::shared_ptr<TERRA> TERRA_frame, int plane_thres) {
        TicToc t_update;
        int update_num = 0;
        int occluded_num = 0;
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < feature_length; i++) {
            *area_points[i] += *TERRA_frame->get_area_points(i);   
        }
        
        std::cout << "\033[32m"; // 设置文本颜色为绿色
        std::cout<<"t_update: "<<t_update.toc()<<std::endl;
        std::cout << "\033[0m";  // 重置文本颜色
        return occluded_num;
    }
    // void visualization() {
    //     draw_ball(feature);
    //     draw_pointcloud_feature();
    //     draw_pointcloud_indice();
    //     draw_pointcloud_normal();
    // }

    void evaluate_degeneracy(){
        Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
        double num = 0;
        for (int i = 0; i < feature_length; i++) {
            if(feature[i] == 0)
                continue;
            covMat = covMat + normals[i] * normals[i].transpose();
            num++;
        }
        covMat = covMat / num;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        // ROS_WARN("degenaracy: %f, %f, %f, the deg direction is %f, %f, %f, the strong dir is %f, %f, %f", 
                //  saes.eigenvalues()[0], saes.eigenvalues()[1], saes.eigenvalues()[2],
                //  saes.eigenvectors().col(0)(0), saes.eigenvectors().col(0)(1), saes.eigenvectors().col(0)(2),
                //  saes.eigenvectors().col(2)(0), saes.eigenvectors().col(2)(1), saes.eigenvectors().col(2)(2));

        num = 0;
        degenaracy_value = saes.eigenvalues();
        degenaracy_direction = saes.eigenvectors();

        for (int i = 0; i < feature_length; i++) {
            if(feature[i] == 0)
                continue;
            double weight = ((1 / saes.eigenvalues()[0] * normals[i].dot(saes.eigenvectors().col(0)))
                           + (1 / saes.eigenvalues()[1] * normals[i].dot(saes.eigenvectors().col(1)))
                           + (1 / saes.eigenvalues()[2] * normals[i].dot(saes.eigenvectors().col(2))));
            Eigen::Vector3d weighted_normal = normals[i] * weight;
            covMat = covMat + weighted_normal * weighted_normal.transpose();
            num += weight*weight;
        }
        covMat = covMat / num;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes2(covMat);
        // ROS_WARN("after degenaracy: %f, %f, %f, the deg direction is %f, %f, %f, the strong dir is %f, %f, %f\n", 
        //          saes2.eigenvalues()[0], saes2.eigenvalues()[1], saes2.eigenvalues()[2],
        //          saes2.eigenvectors().col(0)(0), saes2.eigenvectors().col(0)(1), saes2.eigenvectors().col(0)(2),
        //          saes2.eigenvectors().col(2)(0), saes2.eigenvectors().col(2)(1), saes2.eigenvectors().col(2)(2));
    }

    pcl::PointCloud<PointType>::Ptr get_feature_ball_global() {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < feature_length; i++) {
            ////// downsample
            if(i  % 10 != 0)
                continue;
            if (feature[i] != 0) {
                Eigen::Vector3d point(index_tree->getInputCloud()->points[i].x, 
                                      index_tree->getInputCloud()->points[i].y, 
                                      index_tree->getInputCloud()->points[i].z);
                
                Eigen::Vector3d tf_point = base_pose * point;

                pcl::PointXYZI p;
                p.x = tf_point.x();
                p.y = tf_point.y();
                p.z = tf_point.z();
                p.intensity = feature[i] / feature.maxCoeff();

                cloudout->points.push_back(p);
            }
        }
        return cloudout;
    }

    pcl::PointCloud<PointType>::Ptr get_ballpoint_times_feature() {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < feature_length; i++) {
            Eigen::Vector3d point(index_tree->getInputCloud()->points[i].x, 
                                    index_tree->getInputCloud()->points[i].y, 
                                    index_tree->getInputCloud()->points[i].z);
            point = feature[i] *point;
            pcl::PointXYZI p;
            p.x = point.x();
            p.y = point.y();
            p.z = point.z();
            // p.intensity = feature[i] / feature.maxCoeff();
            cloudout->points.push_back(p);
            
        }
        return cloudout;
    }


    
    //TERRA to pcl::PointXYZI
    void pub_ros_feature(ros::Publisher publisher, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<pcl::PointXYZI>);
        // std::cout<<"feature_id:"<<" ";

        for (int i = 0; i < feature_length; i++) {
            if (feature[i] != 0) {
                // std::cout<<i<<" ";
                pcl::PointXYZI p;
                p.x = normals[i](0);
                p.y = normals[i](1);
                p.z = normals[i](2);
                p.intensity = feature[i];

                cloudout->points.push_back(p);
            }
            else{
                pcl::PointXYZI p;
                p.x = 0;
                p.y = 0;
                p.z = 0;
                p.intensity = 0;

                cloudout->points.push_back(p);
            }            
        }
        // std::cout<<std::endl;

        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
    }

    //pcl::PointXYZINormal to TERRA 

    void from_ros_feature(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
        
        for (int i = 0; i < feature_length; i++) {
            feature[i] = cloud->points[i].intensity;
            normals[i](0) = cloud->points[i].x;
            normals[i](1) = cloud->points[i].y;
            normals[i](2) = cloud->points[i].z;
        }
    }



    Eigen::Isometry3d get_base_pose(){
        return base_pose;
    }

    int get_point_index(int i){ //i is the index of the point in the original cloud
        return indices[i];
    }

    pcl::PointCloud<PointType>::Ptr get_cloud(){ //i is the index of the point in the original cloud
        return cloud;
    }

    pcl::PointCloud<PointType>::Ptr get_area_points(int i){//i is the index of feature
        return area_points[i];
    }
    double get_ballpoint_feature(int i){//i is the index of feature
        return feature[i];
    }
    double get_point_feature(int i){//i is the index of the point in the original cloud
        return get_ballpoint_feature(indices[i]);
    }
    int get_valid_feature_num(){
        int out = 0;
        for(int i = 0; i < feature_length; i++){
            if(feature[i] != 0)
                out++;
        }
        return out;
    }

    int get_valid_normal_num(){
        int out = 0;
        for(int i = 0; i < feature_length; i++){
            if(normals[i](0) != 0 && normals[i](1) != 0 && normals[i](2) != 0)
                out++;
        }
        return out;
    }
    

    


    Eigen::Vector3d get_ballpoint_normal(int i){//i is the index of feature
        return normals[i];
    }
    void set_regist_flag(int i){
        regist_flag[i] = 1;
    }

    void clean_regist_flag(){
        for(int i = 0; i < feature_length; i++){
            regist_flag[i] = 0;
        }
    }

    void draw_ball(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < feature_length; i++) {
            if (feature[i] != 0) {
                pcl::PointXYZI p;
                p.x = index_tree->getInputCloud()->points[i].x;
                p.y = index_tree->getInputCloud()->points[i].y;
                p.z = index_tree->getInputCloud()->points[i].z;

                p.intensity = feature[i] / feature.maxCoeff();

                cloudout->points.push_back(p);
            }
        }
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
    }  

    void draw_feature_global(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < feature_length; i++) {
            ////// downsample
            // if(i  % 10 != 0)
            //     continue;
            if (feature[i] != 0) {
                Eigen::Vector3d point(index_tree->getInputCloud()->points[i].x, 
                                      index_tree->getInputCloud()->points[i].y, 
                                      index_tree->getInputCloud()->points[i].z);
                point = feature[i] *point;
                Eigen::Vector3d tf_point = base_pose * point;

                pcl::PointXYZI p;
                p.x = tf_point.x();
                p.y = tf_point.y();
                p.z = tf_point.z();
                p.intensity = feature[i] / feature.maxCoeff();

                cloudout->points.push_back(p);
            }
        }
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);

    }

    void draw_feature_plane(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < feature_length; i++) {
            std::random_device rd; // Obtain a random number from hardware
            std::mt19937 gen(rd()); // Seed the generator
            std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range
            if(i % 3 != 0)
                continue;
            if (feature[i] != 0) {
                Eigen::Vector3d point(index_tree->getInputCloud()->points[i].x, 
                                      index_tree->getInputCloud()->points[i].y, 
                                      index_tree->getInputCloud()->points[i].z);
                point = feature[i] * point;
                double color = distr(gen);
                pcl::PointXYZI p;
                p.x = point.x();
                p.y = point.y();
                p.z = point.z();
                p.intensity = color;
                cloudout->points.push_back(p);
                double D = - normals[i].dot(point);
                for (double dx = -1.0; dx <= 1.0; dx += 0.5) {
                    for (double dy = -1.0; dy <= 1.0; dy += 0.5) {
                        double z = (-D - normals[i](0) * (dx+point.x()) - normals[i](1) * (dy+point.y())) / normals[i](2);
                        Eigen::Vector3d point_draw(dx+point.x(), dy+point.y(), z);
                        point_draw = (point_draw-point) / (point_draw-point).norm() * abs(dx) + point;
                        p.x = point_draw.x();
                        p.y = point_draw.y();
                        p.z = point_draw.z();
                        p.intensity = color;
                        cloudout->points.push_back(p);  
                    }
                }   
            

            }
        }
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);

    }


    void draw_pointcloud_feature(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < (int)cloud->points.size(); i++) {

            if (feature[indices[i]] != 0) {
                pcl::PointXYZI p;
                p.x = cloud->points[i].x;
                p.y = cloud->points[i].y;
                p.z = cloud->points[i].z;
                p.intensity = feature[indices[i]] / feature.maxCoeff();
                cloudout->points.push_back(p);
            }
        }
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
    }

    void draw_pointcloud_mineigenvalue(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < (int)cloud->points.size(); i++) {

            if (feature[indices[i]] != 0) {
                pcl::PointXYZI p;
                p.x = cloud->points[i].x;
                p.y = cloud->points[i].y;
                p.z = cloud->points[i].z;
                p.intensity = min_eigenvalue[indices[i]];
                cloudout->points.push_back(p);
            }
        }
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
    }

    void draw_pointcloud_indice(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        std::vector<float> random_numbers(feature_length);

        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_real_distribution<> distr(0.0, 1.0); // Define the range

        for (int i = 0; i < feature_length; ++i) {
            random_numbers[i] = distr(gen); // Generate random numbers
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for(int j = 0; j < area_points.size(); j++){
            for (int i = 0; i < (int)area_points[j]->points.size(); i++) {
                pcl::PointXYZI p;
                p.x = area_points[j]->points[i].x;
                p.y = area_points[j]->points[i].y;
                p.z = area_points[j]->points[i].z;
                p.intensity = random_numbers[j];
                cloudout->points.push_back(p);
            }
        } 
        
        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
        // Implement this function to draw the point cloud with indices
    }

    void draw_pointcloud_regist_flag(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        std::vector<float> random_numbers(feature_length);

        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_real_distribution<> distr(0.3, 1.0); // Define the range

        for (int i = 0; i < feature_length; ++i) {
            random_numbers[i] = distr(gen); // Generate random numbers
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<PointType> );
        for (int i = 0; i < (int)cloud->points.size(); i++) {

            if (feature[indices[i]] != 0) {
                pcl::PointXYZI p;
                p.x = cloud->points[i].x;
                p.y = cloud->points[i].y;
                p.z = cloud->points[i].z;
                if(regist_flag[indices[i]] == 1)
                    p.intensity = distr(gen);
                else
                    p.intensity = 0;
                cloudout->points.push_back(p);
            }
        }

        


        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
        // Implement this function to draw the point cloud with indices
    }

    void draw_pointcloud_normal(ros::Publisher publisher, double timestamp, std::string frame_id = "map") {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudout(new pcl::PointCloud<pcl::PointXYZINormal>);
        

        for(int j = 0; j < area_points.size(); j++){
            for (int i = 0; i < (int)area_points[j]->points.size(); i++) {
                if(feature[j] == 0)
                    continue;
                pcl::PointXYZINormal p;
                p.x = area_points[j]->points[i].x;
                p.y = area_points[j]->points[i].y;
                p.z = area_points[j]->points[i].z;
                p.intensity = feature[j];

                p.normal_x = normals[j](0);
                p.normal_y = normals[j](1);
                p.normal_z = normals[j](2);
                cloudout->points.push_back(p);
            }
        } 


        sensor_msgs::PointCloud2 cloudtempmsg;
        pcl::toROSMsg(*cloudout, cloudtempmsg);
        cloudtempmsg.header.stamp =  ros::Time().fromSec(timestamp);
        cloudtempmsg.header.frame_id = frame_id;
        publisher.publish(cloudtempmsg);
    }

private:
    pcl::KdTreeFLANN<PointType>::Ptr index_tree;
    pcl::PointCloud<PointType>::Ptr cloud;
    double timestamp;
    int feature_length;
    Eigen::VectorXd feature;
    Eigen::VectorXd ball_distence;
    std::vector<Eigen::Vector3d> normals;
    Eigen::VectorXd indices;
    std::vector<Eigen::Vector3d> centers;
    std::vector<int> area_point_num;
    std::vector< Eigen::Matrix3d> covMat;
    Eigen::VectorXd regist_flag;
    Eigen::VectorXd min_eigenvalue;
    std::vector<pcl::PointCloud<PointType>::Ptr> area_points;

};
