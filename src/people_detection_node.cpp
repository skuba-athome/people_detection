//
// Created by kandithws on 7/1/2559.
//

#include <cstdlib>
#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include "PeopleDetector.h"
#include "PeopleTracker.h"

#include <people_detection/PersonObject.h>
#include <people_detection/PersonObjectArray.h>
#include <people_detection/ClearPeopleTracker.h>

#define DEFAULT_CLOUD_TOPIC "/camera/depth_registered/points"
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class PeopleDetectionRunner{
    private:
        ros::NodeHandle nh;
        ros::Publisher people_array_pub;
        ros::ServiceServer service;
        PointCloudT::Ptr cloud_obj;
        ros::Subscriber cloub_sub;
        bool new_cloud_available_flag;
        PeopleDetector ppl_detector;
        std::vector<person> world_track_list;



    public:

        PeopleDetectionRunner():
                nh("~"),
                cloud_obj(new PointCloudT)
                {
                    this->new_cloud_available_flag = false;
                    double track_distance;
                    double min_confidence;
                    double min_height;
                    double max_height;
                    double detect_range;
                    double head_min_dist;
                    bool ui_enable;

                    //Init ROS NODE
                    this->cloub_sub = nh.subscribe(DEFAULT_CLOUD_TOPIC, 1, &PeopleDetectionRunner::cloudCallback, this);
                    this->people_array_pub = nh.advertise<people_detection::PersonObjectArray>("peoplearray", 10);
                    this->service = nh.advertiseService("/clearpeopletracker", &PeopleDetectionRunner::cleartrackCallback, this);

                    //INIT ROS PARAM

                    std::string ref_file_path;
                    nh.param<std::string>( "ref_svm_path", ref_file_path, "/trainedLinearSVMForPeopleDetectionWithHOG.yaml");
                    std::string svm_filename = ros::package::getPath("people_detection") + ref_file_path;
                    ROS_INFO( "ref_svm_path: %s", ref_file_path.c_str() );

                    std::string string_intrinsic;
                    nh.param<std::string>( "rgb_intrinsic", string_intrinsic, "525 0.0 319.5 0.0 525 239.5 0.0 0.0 1.0"); //Default = Kinect RGB Intrinsic Params
                    Eigen::Matrix3f rgb_intrinsic = PeopleDetector::IntrinsicParamtoMatrix3f(string_intrinsic);
                    ROS_INFO( "rgb_intrinsic: %s", string_intrinsic.c_str() );

                    nh.param( "detect_range", detect_range, DEFAULT_DETECT_RANGE );
                    ROS_INFO( "detect_range: %lf", detect_range );

                    nh.param( "track_distance", track_distance, DEFAULT_TRACK_DISTANCE );
                    ROS_INFO( "track_distance: %lf", track_distance );

                    nh.param( "min_confidence", min_confidence, DEFAULT_MIN_CONFIDENCE );
                    ROS_INFO( "min_confidence: %lf", min_confidence );

                    nh.param( "min_height", min_height, DEFAULT_MIN_HEIGHT );
                    ROS_INFO( "min_height: %lf", min_height);

                    nh.param( "max_height", max_height, DEFAULT_MAX_HEIGHT );
                    ROS_INFO( "max_height: %lf", max_height);

                    nh.param( "head_min_distance", head_min_dist, DEFAULT_HEAD_MINIMUM_DISTANCE );
                    ROS_INFO( "head_min_distance: %lf", head_min_dist);

                    nh.param( "ui", ui_enable, true);
                    ROS_INFO( "ui_enable: %d", ui_enable);

                    //Init other parameters
                    this->ppl_detector.initPeopleDetector(svm_filename, rgb_intrinsic,min_height, max_height, head_min_dist, detect_range, ui_enable);
                };

        void execute()
        {
            if(this->new_cloud_available_flag)
            {
                std::vector<Eigen::Vector3f> tmp_center_list;
                this->ppl_detector.getPeopleCenter(this->cloud_obj,tmp_center_list);

                this->ppl_detector.viewer.spinOnce();
                this->new_cloud_available_flag = false;
            }

        }

        void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
        {

            pcl::fromROSMsg (*cloud_in, *cloud_obj);
            this->new_cloud_available_flag = true;
            //ROS_INFO("%ld",cloud_obj->size());
            std::cout << "-*********New Cloud*********" << std::endl;
            ROS_INFO("CLOUD LINK = %s",cloud_in->header.frame_id.c_str());
            std::cout << "----------------------------" << std::endl;
        }

        bool cleartrackCallback(people_detection::ClearPeopleTracker::Request &req,
                                people_detection::ClearPeopleTracker::Response &res)
        {
            //Clear World Track List
            //TODO--Change to ActionLib
            /*
            world_track_list.clear();
            lastavailable_id = 0;*/
            return true;
        }

};

int main( int argc, char **argv ) {

    ros::init( argc, argv, "people_detection");
    PeopleDetectionRunner runner;
    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        runner.execute();
        ros::spinOnce();
        loop_rate.sleep();
    }


    return 0;

}

