//
// Created by kandithws on 7/1/2559.
//

#include "PeopleDetector.h"

using namespace Eigen;

//---------------Public---------------------

PeopleDetector::PeopleDetector()
{
    //empty constructor for easily coding purpose
}

void PeopleDetector::initPeopleDetector(std::string svm_filename,Eigen::Matrix3f rgb_intrinsics_matrix, double minheight, double maxheight,
                                         double min_condf, double headmindist, double detectrange)
{
    float voxel_size = 0.06;

    this->heads_minimum_distance = headmindist;
    this->min_height = minheight;
    this->max_height = maxheight;
    this->detect_range = detectrange;
    this->min_confidence = min_condf;

    /*if(viewer_enable)
    {
        this->ui_enable = true;
        this->viewer = pcl::visualization::PCLVisualizer("PCL Viewer");
        this->viewer.setCameraPosition(0,0,-2,0,-1,0,0);
    }
    else
    {
        this->ui_enable = false;
    }*/

    this->person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM
    this->people_detector.setVoxelSize(voxel_size);                        // set the voxel size
    this->people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
    this->people_detector.setClassifier(this->person_classifier);                // set person classifier
    this->people_detector.setHeightLimits((float)min_height, (float)max_height);         // for pcl 1.7.1
    this->people_detector.setMinimumDistanceBetweenHeads((float)heads_minimum_distance);

    this->camera_optical_frame = DEFAULT_CAM_LINK;
    this->robot_frame = DEFAULT_ROBOT_LINK;
}



void PeopleDetector::getPeopleCenter(PointCloudT::Ptr cloud, std::vector<Eigen::Vector3f>& center_list){

    Eigen::VectorXf ground_coeffs = getGroundCoeffs();

    std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

    // Perform people detection on the new cloud:
    this->clusters.clear();
    this->people_detector.setInputCloud(cloud);
    this->people_detector.setGround(ground_coeffs);                    // set floor coefficients
    this->people_detector.compute(clusters);                           // perform people detection

    std::cout << "----------------------test----------------" << std::endl;
    unsigned int k = 0;

    for(std::vector< pcl::people::PersonCluster<PointT> >::iterator it = this->clusters.begin(); it != this->clusters.end(); ++it)
    {
        //std::cout << "Minconfidence = "<< min_confidence << std::endl;
        //std::cout << "get new confidence value = " << it->getPersonConfidence() << std::endl;
        if(it->getPersonConfidence() > this->min_confidence) // draw only people with confidence above a threshold
        {
            k++;
            Eigen::Vector3f temp = it->getTCenter();
            if(temp(2) < this->detect_range)
                center_list.push_back(temp);
            std::cout << "Person " << k << " Position : X = " << temp(0) << " ,Y = " << temp(1) << " ,Z = " << temp(2) << std::endl;
        }
    }
    
}

void PeopleDetector::addnewCloudtoViewer(PointCloudT::Ptr cloud, pcl::visualization::PCLVisualizer& viewer_obj)
{
        viewer_obj.removeAllPointClouds();
        viewer_obj.removeAllShapes();
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
        viewer_obj.addPointCloud<PointT> (cloud, rgb, "input_cloud");
}

void PeopleDetector::drawPeopleDetectBox(pcl::visualization::PCLVisualizer& viewer_obj)
{   
    unsigned int k=0;
    for(std::vector< pcl::people::PersonCluster<PointT> >::iterator it = this->clusters.begin(); it != this->clusters.end(); ++it)
    {
        if(it->getPersonConfidence() > this->min_confidence) // draw only people with confidence above a threshold
        {
        // draw theoretical person bounding box in the PCL viewer:
            it->drawTBoundingBox(viewer_obj, k);
            k++;
        }
    }
}

//--------------------------------------- Static Method ----------------------------------------------------------

//Change Intrinsic Params
 Eigen::Matrix3f PeopleDetector::IntrinsicParamtoMatrix3f(std::string intrinsic_string)
{
    std::vector<float> intrinsic;
    std::istringstream is(intrinsic_string);
    Eigen::Matrix3f rgb_intrinsic;
    double value;
    while( is >> value ) {
        intrinsic.push_back(value);
    }
    if (intrinsic.size() < 9) {
        ROS_WARN("Provided RGB CAM Intrinsic Parameters size less than 3x3, Using Default Value (KINECT)");
        intrinsic.clear();
        rgb_intrinsic << 525, 0.0, 319.5,
                        0.0, 525, 239.5,
                        0.0, 0.0, 1.0;
    }
    else
    {
        rgb_intrinsic << intrinsic[0], intrinsic[1], intrinsic[2],
                intrinsic[3] , intrinsic[4], intrinsic[5],
                intrinsic[6], intrinsic[7], intrinsic[8];
    }

    return rgb_intrinsic;
}



float PeopleDetector::compute_norm3(Eigen::Vector3f A, Eigen::Vector3f B)
{
    float delx2 = (A(0)-B(0))*(A(0)-B(0));
    float dely2 = (A(1)-B(1))*(A(1)-B(1));
    float delz2 = (A(2)-B(2))*(A(2)-B(2));
    return sqrt(delx2+dely2+delz2);
}



//---------------Private-------------------

//get Homogeneous Transform Matrix (4x4)
Eigen::Matrix4f PeopleDetector::getHomogeneousMatrix(std::string input_frame, std::string des_frame)
{
    tf::StampedTransform transform;
    try{
        this->listener.lookupTransform(des_frame, input_frame, ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
    }
    Eigen::Matrix4f T;
    pcl_ros::transformAsMatrix(transform,T);
    return T;
}

Eigen::VectorXf PeopleDetector::getGroundCoeffs()
{
    tf::StampedTransform transform;
    try{
        this->listener.lookupTransform(this->robot_frame, this->camera_optical_frame, ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
    }

    Eigen::Matrix4f T;

    pcl_ros::transformAsMatrix(transform,T);

    Eigen::MatrixXf coeffs(1,4); coeffs << 0, 0, 1, 0;
    Eigen::MatrixXf coeffs_out(1,4);
    coeffs_out = coeffs*T;

    Eigen::Vector4f re_co(4); re_co << coeffs_out(0,0), coeffs_out(0,1), coeffs_out(0,2), coeffs_out(0,3);
    return re_co;
}



void PeopleDetector::extractRGBFromPointCloud (boost::shared_ptr<PointCloudT> input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud)
{
    // Extract RGB information from a point cloud and output the corresponding RGB point cloud
    output_cloud->points.resize(input_cloud->height*input_cloud->width);
    output_cloud->width = input_cloud->width;
    output_cloud->height = input_cloud->height;

    pcl::RGB rgb_point;
    for (int j = 0; j < input_cloud->width; j++)
    {
        for (int i = 0; i < input_cloud->height; i++)
        {
            rgb_point.r = (*input_cloud)(j,i).r;
            rgb_point.g = (*input_cloud)(j,i).g;
            rgb_point.b = (*input_cloud)(j,i).b;
            (*output_cloud)(j,i) = rgb_point;
        }
    }
}

