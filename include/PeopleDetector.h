//
// Created by kandithws on 7/1/2559.
//

#ifndef PEOPLE_DETECTION_PEOPLE_DETECTOR_H
#define PEOPLE_DETECTION_PEOPLE_DETECTOR_H


#include <ros/ros.h>
#include <ros/package.h>
#include <boost/thread/thread.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/people/person_cluster.h>
#include <pcl/people/person_classifier.h>

#include <sstream>
#include <stdlib.h>


#define DEFAULT_FRAME_OUT_CONDITION 12
#define DEFAULT_FRAME_IN_CONDITION 3
#define DEFAULT_FRAME_ENTRY_LIFETIME 5

#define KINECT "kinect"


#define DEFAULT_DETECT_RANGE 3.5
#define DEFAULT_TRACK_DISTANCE 0.3
#define DEFAULT_MIN_CONFIDENCE -1.5

#define DEFAULT_MIN_HEIGHT 0.8
#define DEFAULT_MAX_HEIGHT 2.3

#define DEFAULT_HEAD_MINIMUM_DISTANCE 0.2


#define COLOR_VISUALIZE //Comment this and Remake to turn-off visualizer

#define VISUALIZE_ENABLE 1
#define VISUALIZE_DISABLE 0

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;



class PeopleDetector {

public:
    //Public Parameters
    std::string camera_optical_frame;
    std::string robot_frame;

    //Public Functions
    PeopleDetector();
    void initPeopleDetector(std::string svm_filename,Eigen::Matrix3f rgb_intrinsics_matrix, double minheight, double maxheight,
                             double min_condf, double headmindist, double detect_range);
    void getPeopleCenter(PointCloudT::Ptr cloud, std::vector<Eigen::Vector3f>& center_list );
    void addNewCloudToViewer(PointCloudT::Ptr cloud, pcl::visualization::PCLVisualizer::Ptr viewer_obj);
    void drawPeopleDetectBox(pcl::visualization::PCLVisualizer::Ptr viewer_obj);
    void setRobotFrame(std::string camera_link,std::string robot_base_link);


    //Static Methods
    static Eigen::Matrix3f IntrinsicParamtoMatrix3f(std::string intrinsic_string);

private:
    //Private Parameters
    tf::TransformListener listener;
    Eigen::Matrix3f rgb_intrinsics_matrix;
    pcl::people::PersonClassifier<pcl::RGB> person_classifier;
    pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;
    std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
    //std::vector<Eigen::Vector3f> pp_center_list; //buffer for newly detected ppl center
    double min_confidence;
    double min_height;
    double max_height;
    double detect_range;
    double heads_minimum_distance;
    bool ui_enable;

    //Private Functions
    Eigen::VectorXf getGroundCoeffs();


};


#endif //PEOPLE_DETECTION_PEOPLE_DETECTOR_H
