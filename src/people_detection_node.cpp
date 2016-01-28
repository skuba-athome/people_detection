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
#define DEFAULT_CAM_LINK "camera_rgb_optical_frame"
#define DEFAULT_ROBOT_LINK "base_link"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
const std::string algorithm_name[] = {"Single Nearest Neighbor", "Multi People Nearest Neighbor", "Kalman"};
const std::string frame_count_method_name[] ={"NORMAL","With Frame Count"};



class PeopleDetectionRunner{
    private:
        ros::NodeHandle nh;
        ros::Publisher people_array_pub;
        ros::ServiceServer service;
        PointCloudT::Ptr cloud_obj;
        ros::Subscriber cloub_sub;
        bool new_cloud_available_flag;
        PeopleDetector ppl_detector;
        PeopleTracker ppl_tracker;
        std::vector<person> world_track_list;
        //pcl::visualization::PCLVisualizer viewer;
        pcl::visualization::PCLVisualizer::Ptr viewer;
        bool init_cam_frame;
        std::string camera_frame;
        std::string robot_ref_frame;
        bool ui_enable;
        int get_in_track;
        int get_in_check;
        int out_of_track;
        int track_algorithm;
        int frame_count_method;
        tf::TransformListener listener;


        void publishPersonObjectArray(std::vector<person> &tracklist)
        {
            people_detection::PersonObjectArray pubmsg;
            pubmsg.header.stamp = ros::Time::now();
            pubmsg.header.frame_id = this->robot_ref_frame;

            //Transform Publish point

            Eigen::Matrix4f tfmat = this->getHomogeneousMatrix(this->camera_frame, this->robot_ref_frame);
            for(int i=0 ;i < tracklist.size();i++)
            {
                if(tracklist[i].istrack == true)
                {
                    people_detection::PersonObject pers;
                    Eigen::Vector4f pubpts;
                    pubpts << tracklist[i].points(0),tracklist[i].points(1),tracklist[i].points(2),1.0;
                    pubpts = tfmat*pubpts;

                    pers.personpoints.x = pubpts(0);
                    pers.personpoints.y = pubpts(1);
                    pers.personpoints.z = pubpts(2);

                    pers.id = tracklist[i].id;
                    pubmsg.persons.push_back(pers);
                }
            }
            this->people_array_pub.publish(pubmsg);
         }

        //get Homogeneous Transform Matrix (4x4)
        Eigen::Matrix4f getHomogeneousMatrix(std::string input_frame, std::string des_frame)
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

    public:

        PeopleDetectionRunner():
                nh("~"),
                cloud_obj(new PointCloudT)
                {
                    this->init_cam_frame = false;
                    this->new_cloud_available_flag = false;
                    double track_distance;
                    double min_confidence;
                    double min_height;
                    double max_height;
                    double detect_range;
                    double head_min_dist;
                    std::string ref_file_path;
                    std::string string_intrinsic;
                    //Init ROS NODE
                    this->cloub_sub = nh.subscribe(DEFAULT_CLOUD_TOPIC, 1, &PeopleDetectionRunner::cloudCallback, this);
                    this->people_array_pub = nh.advertise<people_detection::PersonObjectArray>("peoplearray", 1);
                    this->service = nh.advertiseService("/clearpeopletracker", &PeopleDetectionRunner::cleartrackCallback, this);

                    //INIT ROS PARAM
                    nh.param<std::string>( "ref_svm_path", ref_file_path, "/trainedLinearSVMForPeopleDetectionWithHOG.yaml");
                    std::string svm_filename = ros::package::getPath("people_detection") + ref_file_path;
                    ROS_INFO( "ref_svm_path: %s", ref_file_path.c_str() );
                    nh.param<std::string>( "rgb_intrinsic", string_intrinsic, "525 0.0 319.5 0.0 525 239.5 0.0 0.0 1.0");
                    //Default = Kinect RGB Intrinsic Params
                    Eigen::Matrix3f rgb_intrinsic = PeopleDetector::IntrinsicParamtoMatrix3f(string_intrinsic);
                    ROS_INFO( "rgb_intrinsic: %s", string_intrinsic.c_str() );

                    nh.param<std::string>( "robot_base_frame", this->robot_ref_frame, DEFAULT_ROBOT_LINK);
                    ROS_INFO( "robot_base_frame: %s", this->robot_ref_frame.c_str());

                    nh.param( "detect_range", detect_range, DEFAULT_DETECT_RANGE );
                    ROS_INFO( "detect_range: %lf", detect_range );


                    nh.param( "min_confidence", min_confidence, DEFAULT_MIN_CONFIDENCE );
                    ROS_INFO( "min_confidence: %lf", min_confidence );

                    nh.param( "min_height", min_height, DEFAULT_MIN_HEIGHT );
                    ROS_INFO( "min_height: %lf", min_height);

                    nh.param( "max_height", max_height, DEFAULT_MAX_HEIGHT );
                    ROS_INFO( "max_height: %lf", max_height);

                    nh.param( "head_min_distance", head_min_dist, DEFAULT_HEAD_MINIMUM_DISTANCE );
                    ROS_INFO( "head_min_distance: %lf", head_min_dist);

                    nh.param( "ui", this->ui_enable, true);
                    ROS_INFO( "ui_enable: %d", this->ui_enable);

                    nh.param( "track_distance", track_distance, DEFAULT_TRACK_DISTANCE );
                    ROS_INFO( "track_distance: %lf", track_distance );

                    nh.param( "get_in_condition", this->get_in_track, DEFAULT_GET_IN_TRACK_CONDITION);
                    ROS_INFO( "get_in_condition: %d", this->get_in_track );

                    nh.param( "get_in_check", this->get_in_check, DEFAULT_GET_IN_TRACK_CHECK_FRAME );
                    ROS_INFO( "get_in_check: %d", this->get_in_check);

                    nh.param( "out_track_condition", this->out_of_track, DEFAULT_OUT_OF_TRACK_CONDITION );
                    ROS_INFO( "out_track_condition: %d", this->out_of_track);

                    nh.param( "track_algorithm", this->track_algorithm, MULTI_NEAREST_NEIGHBOR_TRACKER );
                    ROS_INFO( "track_algorithm: %s", algorithm_name[this->track_algorithm].c_str());

                    nh.param( "frame_count_method",this->frame_count_method, UPDATE_WITH_FRAME_COUNT);
                    ROS_INFO( "frame_count_method: %s", frame_count_method_name[this->frame_count_method].c_str());



                    //Init People Detector
                    this->ppl_detector.initPeopleDetector(svm_filename, rgb_intrinsic, min_height, max_height,
                                                                                         min_confidence, head_min_dist, detect_range);

                    this->ppl_tracker.setTrackThreshold(track_distance);
                    this->ppl_tracker.setListUpdateConstraints(this->get_in_track, this->get_in_check, this->out_of_track);
                    //Init PCL Viewer
                    if(this->ui_enable)
                    {
                        viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("PCL Viewer"));
                        viewer->setCameraPosition(0,0,-2,0,-1,0,0);
                    }

                    ROS_INFO("-------Complete Initialization--------");

                };

        void execute()
        {
            if((this->new_cloud_available_flag) && (this->init_cam_frame))
            {
                std::vector<Eigen::Vector3f> tmp_center_list;
                this->ppl_detector.getPeopleCenter(this->cloud_obj,tmp_center_list);
                this->ppl_tracker.trackPeople(this->world_track_list, tmp_center_list, this->track_algorithm, this->frame_count_method);

                if(this->ui_enable)
                {
                    this->ppl_detector.addNewCloudToViewer(this->cloud_obj,viewer);
                    this->ppl_detector.drawPeopleDetectBox(viewer);
                    this->ppl_tracker.addTrackerBall(viewer,world_track_list);

                    if(!viewer->wasStopped())
                        viewer->spinOnce();
                    else
                    {
                        ROS_WARN("----Viewer has been Stopped:Abort Viewer Processing----");
                        this->ui_enable = false;
                    }

                }

                this->publishPersonObjectArray(this->world_track_list);
                this->new_cloud_available_flag = false;
            }
        }

        void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
        {
            if(!this->init_cam_frame)
            {
                this->camera_frame = cloud_in->header.frame_id.c_str();
                this->ppl_detector.setRobotFrame(this->camera_frame,this->robot_ref_frame);
                this->init_cam_frame = true;
                ROS_INFO("Camera frame: %s", this->camera_frame.c_str());
                ROS_INFO("-----DONE: INIT ROBOT FRAME----");
            }

            if(this->new_cloud_available_flag)
                return; //Flush Cloud data while still processing

            pcl::fromROSMsg (*cloud_in, *cloud_obj);
            this->new_cloud_available_flag = true;
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

int main( int argc, char **argv )
{
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

