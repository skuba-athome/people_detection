#include <ros/ros.h>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include "athome_msgs/msg/navigation_goal.msg"
#include <geometry_msgs/PointStamped.h>
#include <shape_msgs/Plane.h>
#include <pcl/ros/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/people/person_cluster.h>
#include <pcl/people/head_based_subcluster.h>
#include <pcl/people/person_classifier.h>
#include <visualization_msgs/MarkerArray.h>

#include <stdlib.h>

template <typename T> std::string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); }

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using namespace Eigen;

ros::Publisher marker_pub,pan_tilt_pub,goal_pub;
tf::TransformListener* listener;

bool usedToBeFound = false;
bool isTrackingLost = false;


std::string camera_optical_frame = "/camera_rgb_optical_frame";
std::string robot_camera_frame = "/pan_link";
std::string robot_frame = "/base_link";
std::string world_frame = "/odom";

PointCloud::Ptr cloud_obj (new PointCloud);
bool new_cloud_available_flag = false;
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_in)
{
	pcl::fromROSMsg(*cloud_in,*cloud_obj);
	new_cloud_available_flag = true;
	ROS_INFO("%d",cloud_obj->size());  		
}

bool need_reinit = false; 
void intiCallback (const std_msgs::Bool::ConstPtr& init_sig)
{
	need_reinit = init_sig->data;
	usedToBeFound = false;
	isTrackingLost = false;
}

void extractRGBFromPointCloud (boost::shared_ptr<PointCloud> input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud)
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

void target_visualization(MatrixXf pos, float height)
{
	visualization_msgs::Marker marker;
	marker.header.frame_id = world_frame;
	marker.header.stamp = ros::Time::now();
	marker.ns = "target_shapes";
  marker.id = 0;
	//marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::ADD;
    
  // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
  marker.pose.position.x = pos(0,0);
  marker.pose.position.y = pos(1,0);
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

	//marker.text = tostr(height);
  // Set the scale of the marker -- 1x1x1 here means 1m on a side
  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = height;

  // Set the color -- be sure to set alpha to something non-zero!
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;

  marker.lifetime = ros::Duration();

	visualization_msgs::Marker marker_text = marker;
	marker_text.id = 1;
	marker_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker_text.pose.position.z = 2.2;
	marker_text.scale.z = 0.4;
	marker_text.text = tostr(height);

	visualization_msgs::MarkerArray pubMark;
	pubMark.markers.push_back(marker);
	pubMark.markers.push_back(marker_text);
    // Publish the marker
	marker_pub.publish(pubMark);
}

float getCameraHeight()
{
	  tf::StampedTransform transform;
    try{
      listener->lookupTransform(robot_frame, robot_camera_frame, ros::Time(0), transform);
			return transform.getOrigin().z();
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
			ROS_ERROR("Invalid camera height");
    }
		return -10;
}

Eigen::VectorXf getGroundCoeffs()
{
	tf::StampedTransform transform;
	try{
		listener->lookupTransform(robot_frame, camera_optical_frame, ros::Time(0), transform);
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

//get Homogeneous Transform Matrix (4x4)
Eigen::Matrix4f getHomogeneousMatrix(std::string input_frame,std::string des_frame)
{
	tf::StampedTransform transform;
	try{
		listener->lookupTransform(des_frame, input_frame, ros::Time(0), transform);
	}
	catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
	}	
		
	Eigen::Matrix4f T;
	pcl_ros::transformAsMatrix(transform,T);

	return T;
}

Eigen::Vector3f getFrameOrigin(std::string ref_frame,std::string des_frame)
{
	tf::StampedTransform transform;
	try{
		listener->lookupTransform(des_frame, ref_frame, ros::Time(0), transform);
	}
	catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
	}
	Eigen::Vector3f T;
	T << transform.getOrigin().x(),transform.getOrigin().y(),transform.getOrigin().z();
	return T;	
}

bool doTracking(std::vector<Eigen::Vector3f> pp_center_list, float disTH, Eigen::Vector3f &target_world)
{
	static MatrixXf pp_center_world_last(4,1);
	static bool init = false;

//-------------fill in homogeneous matrix-----------------//
	Matrix4f bot2worldTF = getHomogeneousMatrix(robot_frame,world_frame);
	Matrix4f cam2botTF = getHomogeneousMatrix(camera_optical_frame,robot_frame);
	Matrix4f cam2worldTF = bot2worldTF*cam2botTF;
//--------------------------------------------------------//
    std::cout << "pp_center_world_last" << pp_center_world_last << std::endl;
	if(!init || need_reinit)
	{
		MatrixXf init_point_bot(4,1); init_point_bot << 1.0,0.0,1.0,1.0;
		pp_center_world_last = bot2worldTF*init_point_bot;
		//std::cout << pp_center_world_last << std::endl;
		init = true;
		need_reinit = false;
	}

	float min_dist = 1000.0f;
	bool isFound = false;
	MatrixXf pp_center_world = pp_center_world_last;

    //std::cout << "pp_center_world_last " << pp_center_world_last << std::endl;   


	for(int k = 0; k<pp_center_list.size(); k++)
	{
		MatrixXf pp_center_cam(4,1); pp_center_cam << pp_center_list[k](0),pp_center_list[k](1),pp_center_list[k](2), 1;
		MatrixXf pp_center_world_tmp = cam2worldTF*pp_center_cam;
        //pp_center_world_tmp(0)*=-1;   
        //pp_center_world_tmp(1)*=-1;
        //std::cout << "pp_center_world "<< pp_center_world_tmp << std::endl;   
        float dist = (pp_center_world_tmp - pp_center_world_last).squaredNorm();
        std::cout << "Dist "  << dist << std::endl ;
		if(dist < disTH && dist<min_dist)
		{
			isFound = true;	
			min_dist = dist;
			pp_center_world = pp_center_world_tmp;
		}
	}

	pp_center_world_last = pp_center_world;
	//std::cout << "people pose:\n" << pp_center_world << "\nwith dis: " << min_dist << std::endl;
	target_world << pp_center_world(0,0), pp_center_world(1,0), pp_center_world(2,0);
	//Matrix4f world2botTF = getHomogeneousMatrix(world_frame,robot_frame);
//std::cout << target_world << std::endl;	


	return isFound;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "people_detection");
  ros::NodeHandle n;
  ros::Subscriber cloub_sub = n.subscribe("camera/depth_registered/points", 1, cloudCallback);

	ros::Subscriber	init_sub = n.subscribe("/follow/init", 1, intiCallback);
	pan_tilt_pub = n.advertise<geometry_msgs::Quaternion>("pan_tilt_main_state", 1);
	goal_pub = n.advertise<lumyai_navigation_msgs::NavGoalMsg>("/follow/point", 1);
    marker_pub = n.advertise<visualization_msgs::MarkerArray>("target_pose", 1);


  listener = new tf::TransformListener();

  // Algorithm parameters:
  float voxel_size = 0.06;
  float min_confidence = -1.5; // -1.5
  float min_height = 0.8;
  float max_height = 2.3;
//===================================================================================
  // set default values for optional parameters:
  int min_points = 30;     // this value is adapted to the voxel size in method "compute"
  int max_points = 5000;   // this value is adapted to the voxel size in method "compute"
  float heads_minimum_distance = 0.3;
//===================================================================================

	std::string svm_filename = "/home/skuba/skuba_athome/people_detection/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
	Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 131.25, 0.0, 79.5, 0.0, 131.25, 59.5, 0.0, 0.0, 1.0;//525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  // Initialize classifier for people detection:  
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

	ros::Rate loop_rate(10);
  while (ros::ok())
  {
		if(new_cloud_available_flag)
		{
			VectorXf ground_coeffs = getGroundCoeffs();
			//std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

			PointCloud::Ptr cloud (new PointCloud);
			pcl::copyPointCloud<PointT, PointT>(*cloud_obj, *cloud);
			new_cloud_available_flag = false;
			
/*
			if(cloud_obj->size()==0){
				continue;
			}
*/
	
      // Perform people detection on the new cloud:
        std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters

	    // Adapt thresholds for clusters points number to the voxel size:
  	  max_points = int(float(max_points) * std::pow(0.06/voxel_size, 2));
  	  if (voxel_size > 0.06)
  	  min_points = int(float(min_points) * std::pow(0.06/voxel_size, 2));

  		// Fill rgb image:
  		pcl::PointCloud<pcl::RGB>::Ptr rgb_image(new pcl::PointCloud<pcl::RGB>);
  		extractRGBFromPointCloud(cloud, rgb_image);          // fill RGB pointcloud

  		// Voxel grid filtering:
  		PointCloud::Ptr cloud_filtered(new PointCloud);
  		pcl::VoxelGrid<PointT> voxel_grid_filter_object;
  		voxel_grid_filter_object.setInputCloud(cloud);
  		voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
  		voxel_grid_filter_object.filter (*cloud_filtered);

  		// Ground removal and update:
  		pcl::IndicesPtr inliers(new std::vector<int>);
  		pcl::SampleConsensusModelPlane<PointT>::Ptr ground_model(new pcl::SampleConsensusModelPlane<PointT>(cloud_filtered));
  		ground_model->selectWithinDistance(ground_coeffs, voxel_size*3.5, *inliers);
  		PointCloud::Ptr no_ground_cloud(new PointCloud);
  		pcl::ExtractIndices<PointT> extract;
  		extract.setInputCloud(cloud_filtered);
  		extract.setIndices(inliers);
  		extract.setNegative(true);
  		extract.filter(*no_ground_cloud);

  		// Euclidean Clustering:
  		std::vector<pcl::PointIndices> cluster_indices;
  		typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  		tree->setInputCloud(no_ground_cloud);
  		pcl::EuclideanClusterExtraction<PointT> ec;
  		ec.setClusterTolerance(2 * 0.06);
  		ec.setMinClusterSize(min_points);
		  ec.setMaxClusterSize(max_points);
		  ec.setSearchMethod(tree);
  		ec.setInputCloud(no_ground_cloud);
  		ec.extract(cluster_indices);

  		// Head based sub-clustering //
  		pcl::people::HeadBasedSubclustering<PointT> subclustering;
  		subclustering.setInputCloud(no_ground_cloud);
  		subclustering.setGround(ground_coeffs);
  		subclustering.setInitialClusters(cluster_indices);
  		subclustering.setHeightLimits(min_height, max_height);
  		subclustering.setMinimumDistanceBetweenHeads(heads_minimum_distance);
  		subclustering.setSensorPortraitOrientation(false);
  		subclustering.subcluster(clusters);

			std::vector<Eigen::Vector3f> pp_center_list;
			//static bool isTrackingLost = false;

/*
// Draw cloud and people bounding boxes in the viewer:
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
      viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
unsigned int k = 0;*/

			//edit by Win
			printf("size of cluster : %d\n",clusters.size());
			if(clusters.size() == 0 )
				continue;


			for(typename std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
  		{

// draw theoretical person bounding box in the PCL viewer:
          //it->drawTBoundingBox(viewer, k);
          //k++;
				if(true || isTrackingLost)
				{
					for(typename std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
					{
						//Evaluate confidence for the current PersonCluster:
						Eigen::Vector3f centroid = rgb_intrinsics_matrix * (it->getTCenter());
						centroid /= centroid(2);
						Eigen::Vector3f top = rgb_intrinsics_matrix * (it->getTTop());
						top /= top(2);
						Eigen::Vector3f bottom = rgb_intrinsics_matrix * (it->getTBottom());
						bottom /= bottom(2);
						it->setPersonConfidence(person_classifier.evaluate(rgb_image, bottom, top, centroid, rgb_intrinsics_matrix, false));
printf("getPerConfidence %f\n",it->getPersonConfidence());
						if(it->getPersonConfidence() > min_confidence)
						{
							pp_center_list.push_back(it->getTCenter());
						}
					}
				}
				else pp_center_list.push_back(it->getTCenter());
			}
		
			bool isFound;
			Eigen::Vector3f pp_pose_world;
			if(isTrackingLost) isFound = doTracking(pp_center_list,1.0,pp_pose_world);
			else isFound = doTracking(pp_center_list,0.35,pp_pose_world);
			
			//static bool usedToBeFound = false;

			static unsigned int lost_count = 0;

			if(isFound)
			{
				usedToBeFound = true;
				isTrackingLost = false;
				lost_count = 0;
			}
			else
			{
				if(!usedToBeFound) std::cout << "Waiting for initialization" << std::endl;
				else
				{
					lost_count++;
					if(lost_count > 3)
					{
						isTrackingLost = true;
						std::cout << ">>>>>>>>>>Lost Master<<<<<<<<<" << std::endl;
						//edit by dear
						//    usedToBeFound = false;
					}
				}
			}

			/*lumyai_navigation_msgs::NavGoalMsg goal_pose;
			if(isTrackingLost) goal_pose.text_msg = "lost";
			else goal_pose.text_msg = "tracking";
			goal_pose.ref_frame = "absolute";
			goal_pose.pose2d.x = pp_pose_world(0);
			goal_pose.pose2d.y = pp_pose_world(1);
			Eigen::Vector3f vec_dir = pp_pose_world - getFrameOrigin(world_frame,robot_frame);
			goal_pose.pose2d.theta = atan2(vec_dir(1),vec_dir(0));
			goal_pub.publish(goal_pose);*/
			
			
	
			Matrix4f world2botTF = getHomogeneousMatrix(world_frame,robot_frame);
			MatrixXf pp_pose_world_tmp(4,1); pp_pose_world_tmp << pp_pose_world(0),pp_pose_world(1),pp_pose_world(2), 1;
			MatrixXf pp_center_bot = world2botTF*pp_pose_world_tmp;
            //pp_center_bot(0)*=-1; //Edit Tf  
			//pp_center_bot(1)*=-1; //Edit Tf
            ////std::cout << pp_center_bot << std::endl;

			static float pan_ang_filter = 0.0f;
			float pan_ang = atan2(pp_center_bot(1,0),pp_center_bot(0,0));
			pan_ang_filter = pan_ang_filter + 0.8f*(pan_ang - pan_ang_filter);
	
			static bool prepstate = false;
			pan_tilt_pub.publish(tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, pan_ang_filter));
			/*if(prepstate == false)
			{
				if(pan_ang > 0.0 && pan_ang > 2.3 && pan_ang < 2.7) 
				{
					ROS_WARN("prepare to turn left");
					system("espeak -ven+f5 -s 120 'prepare to turn left'");
					prepstate = true;
				}
				
			}
			else
			{
				if(pan_ang > 3.0) 
				{
					ROS_WARN("turn left now");
					system("espeak -ven+f5 -s 120 'turn left now'");
					prepstate = false;
				}
				else if(pan_ang < -2.7)
				{
					ROS_WARN("turn right now");
					prepstate = false;
				}
				
			}*/
			//else if(pan_ang_filter< 0.0 && pan_ang_filter < -2.0 && pan_ang_filter>
			ROS_WARN("pan filter : %f", pan_ang_filter);
            std::cout << "IsTracking ::" << isTrackingLost << std::endl;  //dear
			lumyai_navigation_msgs::NavGoalMsg goal_pose;
			if(true && isTrackingLost) {
				goal_pose.text_msg = "lost";
				goal_pose.ref_frame = "absolute";
				goal_pose.pose2d.x = pp_pose_world(0);
				goal_pose.pose2d.y = pp_pose_world(1);
				Eigen::Vector3f vec_dir = pp_pose_world - getFrameOrigin(world_frame,robot_frame);
				goal_pose.pose2d.theta = atan2(vec_dir(1),vec_dir(0));
			}
			else
			{
				goal_pose.text_msg = "tracking";
				goal_pose.ref_frame = "relative";
				goal_pose.pose2d.x = pp_center_bot(0);
				goal_pose.pose2d.y = pp_center_bot(1);
				goal_pose.pose2d.theta = atan2(pp_center_bot(1),pp_center_bot(0));
			}
            //std::cout << "Goal_pose :: " << goal_pose << std::endl ; 
			goal_pub.publish(goal_pose);
			
			
			//unsigned int k = 0;
  		/*for(typename std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
  		{
    		//Evaluate confidence for the current PersonCluster:
    		Eigen::Vector3f centroid = rgb_intrinsics_matrix * (it->getTCenter());
    		centroid /= centroid(2);
    		Eigen::Vector3f top = rgb_intrinsics_matrix * (it->getTTop());
    		top /= top(2);
    		Eigen::Vector3f bottom = rgb_intrinsics_matrix * (it->getTBottom());
    		bottom /= bottom(2);
    		it->setPersonConfidence(person_classifier.evaluate(rgb_image, bottom, top, centroid, rgb_intrinsics_matrix, false));

				if(it->getPersonConfidence() > min_confidence)// || it->getDistance() <1.3f)
				{
					k++;
					pp_center.push_back(it->getTCenter());
					pp_height.push_back(it->getHeight());
					Vector3f pp_pos_tmp = it->getTCenter();
					//std::cout << it->getTCenter() << "#######" <<std::endl;
					//target_visualization(it->getTCenter(), it->getHeight());
					ROS_INFO("Detecting people %d whose center is at:%.2f, %.2f, %.2f #######################", k, pp_pos_tmp(2), -pp_pos_tmp(0), -pp_pos_tmp(1));
				}
  		}*/

			//doTracking(pp_center, pp_height);

			/*int k = 0;
			PointCloud::Ptr cloud_cluster(new PointCloud);
      for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
      {


        if(it->getPersonConfidence() > min_confidence)
        {

					///////////////////////////////
					
					for (std::vector<int>::iterator pit = it->getIndices().indices.begin(); pit != it->getIndices().indices.end(); ++pit)
					{
						cloud_cluster->points.push_back (no_ground_cloud->points[*pit]);					
					}
			
					
					///////////////////////////////////////////
          Eigen::Vector3f pp_center = it->getTCenter();
					target_visualization(pp_center, it->getPersonConfidence());
					ROS_INFO("People %d center:%.2f, %.2f, %.2f #######################", k, pp_center(0), pp_center(1), pp_center(2));
					k++;
        }
      }*/

			/*cloud_cluster->width = cloud_cluster->points.size ();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;
			cloud_cluster->header = no_ground_cloud->header;

			sensor_msgs::PointCloud2 cloud_out;
			pcl::toROSMsg(*cloud_cluster,cloud_out);
 			cloud_out.header.stamp = ros::Time::now();
			cloud_pub.publish(cloud_out);*/

			
	
			loop_rate.sleep();	
		}
	ros::spinOnce();
	}
  return 0;
}


/*void doTracking(std::vector<Eigen::Vector3f> pp_center_list, std::vector<float> pp_height_list)
{
	Matrix4f bot2worldTF = getHomogeneousMatrix(robot_frame,world_frame);
	Matrix4f cam2botTF = getHomogeneousMatrix(camera_optical_frame,robot_frame);
	Matrix4f cam2worldTF = bot2worldTF*cam2botTF;
	static MatrixXf pp_center_world_last(4,1);
	static bool init = false;
	static bool init_height = false;
	static float pp_height_last = 0.5;
	if(!init)
	{
		MatrixXf init_point_bot(4,1); init_point_bot << 1.0,0.0,1.0,1.0;
		pp_center_world_last = bot2worldTF*init_point_bot;
		init = true;
	}

	//std::cout << pp_center_world_last << "-------------" << std::endl;
	float min_dist = 1000.0f;
	
	MatrixXf pp_center_world = pp_center_world_last;
	float pp_height = pp_height_last;
	for(int k = 0; k<pp_center_list.size(); k++)
	{
		float pp_height_tmp = pp_height_list[k];
		MatrixXf pp_center_cam(4,1); pp_center_cam << pp_center_list[k](0),pp_center_list[k](1),pp_center_list[k](2), 1;
		MatrixXf pp_center_world_tmp = cam2worldTF*pp_center_cam;
		float dist = (pp_center_world_tmp - pp_center_world_last).squaredNorm();
		if(dist < 0.3f)
		{
			if(!init_height)
			{
				pp_height_last = pp_height_tmp;
				init_height = true;	
			}
			
			if(init_height && fabs(pp_height_tmp - pp_height_last) < 0.1 && dist<min_dist)
			{	
				min_dist = dist;
				pp_center_world = pp_center_world_tmp;
				pp_height = pp_height_tmp;
			}
		}
	}

	std::cout << "people pose:\n" << pp_center_world << "\nwith dis: " << min_dist << std::endl;

	target_visualization(pp_center_world, pp_height);
	pp_center_world_last = pp_center_world;
	pp_height_last = pp_height;
}*/

/*Eigen::VectorXf ground_coeffs(4,1);
void groundcoeffsCallback(const shape_msgs::Plane::ConstPtr& coeffs_in)
{
	tf::StampedTransform transform;
	try{
		listener->lookupTransform(robot_camera_frame, camera_optical_frame, ros::Time(0), transform);
	}
	catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
	}	
		
	Eigen::Matrix4f T;
	pcl_ros::transformAsMatrix(transform,T);

	Eigen::MatrixXf coeffs(1,4); coeffs << coeffs_in->coef[0], coeffs_in->coef[1], coeffs_in->coef[2], coeffs_in->coef[3];
	Eigen::MatrixXf coeffs_out(1,4);
	coeffs_out = coeffs*T;
	ground_coeffs(0) = coeffs_out(0,0);
	ground_coeffs(1) = coeffs_out(0,1);
	ground_coeffs(2) = coeffs_out(0,2);
	ground_coeffs(3) = coeffs_out(0,3);
	std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;
}*/

/*void doTracking(std::vector<Eigen::Vector3f> pp_center_list, std::vector<float> pp_height_list)
{
	static MatrixXf pp_center_world_last(4,1);
	static bool init = false;

//-------------fill in homogeneous matrix-----------------//
	Matrix4f bot2worldTF = getHomogeneousMatrix(robot_frame,world_frame);
	Matrix4f cam2botTF = getHomogeneousMatrix(camera_optical_frame,robot_frame);
	Matrix4f cam2worldTF = bot2worldTF*cam2botTF;
//--------------------------------------------------------//

	if(!init)
	{
		MatrixXf init_point_bot(4,1); init_point_bot << 1.0,0.0,1.0,1.0;
		pp_center_world_last = bot2worldTF*init_point_bot;
		init = true;
	}

	float min_dist = 1000.0f;
	MatrixXf pp_center_world = pp_center_world_last;
	for(int k = 0; k<pp_center_list.size(); k++)
	{
		MatrixXf pp_center_cam(4,1); pp_center_cam << pp_center_list[k](0),pp_center_list[k](1),pp_center_list[k](2), 1;
		MatrixXf pp_center_world_tmp = cam2worldTF*pp_center_cam;
		float dist = (pp_center_world_tmp - pp_center_world_last).squaredNorm();
		if(dist < 0.3f && dist<min_dist)
		{
				min_dist = dist;
				pp_center_world = pp_center_world_tmp;
		}
	}

	std::cout << "people pose:\n" << pp_center_world << "\nwith dis: " << min_dist << std::endl;

	target_visualization(pp_center_world, 1);
	pp_center_world_last = pp_center_world;
}*/
