#include <ros/ros.h>
#include <ros/package.h>

#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
//#include "athome_msgs/msg/navigation_goal.msg"
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <shape_msgs/Plane.h>
#include <visualization_msgs/MarkerArray.h>

//#include <pcl/ros/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

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

#include <people_detection/PersonObject.h>
#include <people_detection/PersonObjectArray.h>

#include <sstream>
#include <stdlib.h>

#define FRAME_OUT_CONDITION 12
#define FRAME_IN_CONDITION 3
#define FRAME_ENTRY_TRACKLIST 5

#define COLOR_VISUALIZE

template <typename T> std::string tostr(const T& t) { std::ostringstream os; os<<t; return os.str(); }

//typedef pcl::PointXYZRGB PointT;
//typedef pcl::PointCloud<PointT> PointCloud;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace Eigen;

//===================================================================================
   // Algorithm parameters:
  float voxel_size = 0.06;
  float min_confidence = -1.5; // -1.5
  float min_height = 0.8;
  float max_height = 2.3;
  float detect_length = 3.5;
  // set default values for optional parameters:
  int min_points = 30;     // this value is adapted to the voxel size in method "compute"
  int max_points = 5000;   // this value is adapted to the voxel size in method "compute"
  float heads_minimum_distance = 0.2;

//===================================================================================



pcl::visualization::PCLVisualizer viewer("PCL Viewer");
enum { COLS = 640, ROWS = 480 };
//ros::Publisher marker_pub,pan_tilt_pub,goal_pub;
//tf::TransformListener* listener;

bool usedToBeFound = false;
bool isTrackingLost = false;


/*std::string camera_optical_frame = "/camera_rgb_optical_frame";
std::string robot_camera_frame = "/pan_link";
std::string robot_frame = "/base_link";
std::string world_frame = "/odom";*/

PointCloudT::Ptr cloud_obj (new PointCloudT);
bool new_cloud_available_flag = false;


typedef struct{
  Eigen::Vector3f points;
  #ifdef COLOR_VISUALIZE
  Eigen::Vector3f color;
  #endif
  int id;
  int framesage;
  int frameincond;
  int framelostlifetime;
  bool istrack;
}person;


//std::vector<people_detection::PersonObject> world_track_list;
std::vector<person> world_track_list;

int lastavailable_id =0;



void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_in)
{
	
	pcl::fromROSMsg (*cloud_in, *cloud_obj);

	new_cloud_available_flag = true;
	//ROS_INFO("%ld",cloud_obj->size());
  std::cout << "----------------------------" << std::endl; 		
}

struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloudT::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

void extractRGBFromPointCloud (boost::shared_ptr<PointCloudT> input_cloud, pcl::PointCloud<pcl::RGB>::Ptr& output_cloud)
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

Eigen::Vector3f generateTrackerColor()
{
  Eigen::Vector3f color;
  float r = ((double) rand() / (RAND_MAX));
  float g = ((double) rand() / (RAND_MAX));
  float b = ((double) rand() / (RAND_MAX));
  color(0) = r;
  color(1) = g;
  color(2) = b;
  return color;
}

float compute_norm3(Eigen::Vector3f A, Eigen::Vector3f B)
{
  float delx2 = (A(0)-B(0))*(A(0)-B(0));
  float dely2 = (A(1)-B(1))*(A(1)-B(1));
  float delz2 = (A(2)-B(2))*(A(2)-B(2));   
  return sqrt(delx2+dely2+delz2);
}




bool doMultiple_Tracking(std::vector<Eigen::Vector3f> &pp_newcenter_list,std::vector<person> &world, float disTH)
{
  //TODO--Convert ALL INPUT and OUTPUT to WORLD_FRAME 

  std::vector<person> world_temp(world);

  if(!world_temp.empty())
  {
      //Do tracking Find nearest neighbor replace in same index
      MatrixXf nn_matching_table(world_temp.size(),pp_newcenter_list.size());
      //pp_newcenter_temp << pp_newcenter_list[i](0),pp_newcenter_list[i](1),pp_newcenter_list[i](2), 1;
      while(!pp_newcenter_list.empty())
      {
        //float min = ( world_temp[0].points - pp_newcenter_list[0] ).squaredNorm();
         float min = compute_norm3(world_temp[0].points, pp_newcenter_list[0]);
        int   index[] = {0,0};
        //cout << "size world = " << world_temp.size() << endl << "size center = " << pp_newcenter_list.size() <<endl;
        
        for(int i=0; i< world_temp.size() ; i++)
        {
          for(int j=0; j < pp_newcenter_list.size();j++ )
          {
           // nn_matching_table(i,j) = ( world_temp[i].points - pp_newcenter_list[j] ).squaredNorm();
            nn_matching_table(i,j) = compute_norm3(world_temp[i].points, pp_newcenter_list[j]);
            if(nn_matching_table(i,j) < min )
            {
              //update min finding nearest neighbour
              index[0] = i;
              index[1] = j;
              min = nn_matching_table(i,j);
            }
          }
        }
        //select match
        cout <<"--Matiching Table--" << endl;
        for(int i =0 ; i < world_temp.size() ; i++)
        {
          for(int j =0 ; j < pp_newcenter_list.size() ; j++)
          {
            cout << nn_matching_table(i,j) << '\t';
          }
          cout << endl;
        }
        cout << endl;

       
        if((min < disTH) && (!world_temp.empty())) //still track
        {
          cout << "min => " << min << endl;
          for(int k = 0; k < world.size() ; k++)
          { 
            if(world[k].id == world_temp[index[0]].id )
            {
              world[k].points = pp_newcenter_list[index[1]];
              world[k].id = world_temp[index[0]].id;
              world[k].framelostlifetime = FRAME_OUT_CONDITION;
              cout << "Updated Track id --> " <<  world[k].id << endl;
            }
          }
          world_temp.erase(world_temp.begin() + index[0]);
          pp_newcenter_list.erase(pp_newcenter_list.begin() + index[1]);

         // cout << "delete index : " << endl << "Worldtemp = "   << index[0]<< " ,newcenter = " << index[1] << endl;
        }
        //find all nearest neighbour matching
        else //2 points too far 
        {
          //append to be newpoint to add to world
          person temp;
          temp.points = pp_newcenter_list[index[1]] ;
          temp.id = lastavailable_id++;
          temp.framesage = 0;
          temp.framelostlifetime = FRAME_OUT_CONDITION;
          temp.frameincond = FRAME_IN_CONDITION;
          temp.istrack = false;
          #ifdef COLOR_VISUALIZE
          temp.color = generateTrackerColor();
          #endif
          world.push_back(temp);
          //remove from list to calculate
          if(!world_temp.empty())
          {
            world_temp.erase(world_temp.begin() + index[0]);
          }  
          pp_newcenter_list.erase(pp_newcenter_list.begin() + index[1]);
        }
      }

      //Count lost tracking
      if(!world_temp.empty())
      {
        for (int i=0; i < world_temp.size();i++)
        {
          for(int j=0 ; j < world.size();j++)
          {
            if(world[j].istrack == true)
            {
              //if(world[j].id == world_temp[i].id) ++world[j].framescount;
              if(world[j].id == world_temp[i].id) --world[j].framelostlifetime;
            }
            else if(world[j].istrack == false)
            {
              if(world[j].id == world_temp[i].id) --world[j].frameincond;
            }
   
          }  
        }
      }
      
    }
    else
    {
      //No one is tracked from the last frame reinit tracker list
      for(int i = 0 ; i < pp_newcenter_list.size();i++)
      {
        person temp;
        temp.points = pp_newcenter_list[i] ;
        temp.id = lastavailable_id++;
        temp.framesage = 0;
        temp.framelostlifetime = FRAME_OUT_CONDITION;
        temp.frameincond = FRAME_IN_CONDITION;
        temp.istrack = false;
        #ifdef COLOR_VISUALIZE
        temp.color = generateTrackerColor();
        #endif
        world.push_back(temp);
      }
    }
}

void checktracklist(std::vector<person> &tracklist)
{
  //Remove or movein person
  for(int i=0 ; i < tracklist.size() ; i++ )
  { 
    //Remove Person
    if(tracklist[i].framelostlifetime <= 0)
    {
      tracklist.erase(tracklist.begin() + i);
    }
    //New Person checkframe
    if(tracklist[i].istrack == false)
    {
      //Lost Track of new entry frame
      if( (tracklist[i].framesage >= FRAME_ENTRY_TRACKLIST))
      {
        if(tracklist[i].frameincond <= 0)
        {
          tracklist.erase(tracklist.begin() + i);
        }
        else
        {
          tracklist[i].istrack = true;
        }
      }
      else
      {
        tracklist[i].framesage++;
      }
    }
    else
    {
      tracklist[i].framesage++;
    }
  }
}


         
void classifyperson(PointCloudT::Ptr &cloud,std::vector< pcl::people::PersonCluster<PointT> > &clusters,Eigen::VectorXf &ground_coeffs,pcl::PointCloud<pcl::RGB>::Ptr &rgb_image)
{
  

  //pcl::PointCloud<pcl::RGB>::Ptr rgb_image(new pcl::PointCloud<pcl::RGB>);
  rgb_image = pcl::PointCloud<pcl::RGB>::Ptr(new pcl::PointCloud<pcl::RGB>);
  extractRGBFromPointCloud(cloud, rgb_image);

  // Voxel grid filtering:
  PointCloudT::Ptr cloud_filtered(new PointCloudT);
  pcl::VoxelGrid<PointT> voxel_grid_filter_object;
  voxel_grid_filter_object.setInputCloud(cloud);
  voxel_grid_filter_object.setLeafSize (voxel_size, voxel_size, voxel_size);
  voxel_grid_filter_object.filter (*cloud_filtered);

  // Ground removal and update:
  pcl::IndicesPtr inliers(new std::vector<int>);
  pcl::SampleConsensusModelPlane<PointT>::Ptr ground_model(new pcl::SampleConsensusModelPlane<PointT>(cloud_filtered));
  ground_model->selectWithinDistance(ground_coeffs, voxel_size*3.5, *inliers);
  PointCloudT::Ptr no_ground_cloud(new PointCloudT);
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

  //update ground_coeff
  //ground_coeffs = people_detector.getGround();
}





int main(int argc, char **argv)
{
  ros::init(argc, argv, "people_detection");
  ros::NodeHandle n;
  ros::Subscriber cloub_sub = n.subscribe("camera/depth_registered/points", 1, cloudCallback);
  //ros::Subscriber cloub_sub = n.subscribe("/depth_registered/depth_registered/points", 1, cloudCallback); 
 /*Save detect point to file*/
 /* ofstream myfile;
  std::string writefilename = ros::package::getPath("people_detection") + "/sandbox/data1people2.txt";
  myfile.open(writefilename.c_str()); */

	//std::string svm_filename = "/home/skuba/skuba_athome/people_detection/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
	std::string svm_filename = ros::package::getPath("people_detection") + "/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  std::cout << "svm_filename : " << svm_filename << std::endl; 
	Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0;// 650.577610157369, 0.0, 331.019280291833, 0.0, 649.940553093797, 258.968249986678, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics//131.25, 0.0, 79.5, 0.0, 131.25, 59.5, 0.0, 0.0, 1.0;

  // Adapt thresholds for clusters points number to the voxel size:
  max_points = int(float(max_points) * std::pow(0.06/voxel_size, 2));
  if (voxel_size > 0.06)
  min_points = int(float(min_points) * std::pow(0.06/voxel_size, 2));


  ros::Rate loop_rate_wait(4);

  while(!new_cloud_available_flag)
  	{
  		std::cout << "Waiting for first Pointcloud for initiate ground" << std::endl;
  		ros::spinOnce();
  		loop_rate_wait.sleep();
    }

  		// Display pointcloud:
       new_cloud_available_flag = false;
       PointCloudT::Ptr cloud (new PointCloudT);
			pcl::copyPointCloud<PointT, PointT>(*cloud_obj, *cloud);

  		pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  		viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  		viewer.setCameraPosition(0,0,-2,0,-1,0,0);
   
  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  // Spin until 'Q' is pressed:
  viewer.spin();
  std::cout << "done." << std::endl;

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);
  


  // Initialize classifier for people detection:  
   pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // // for pcl 1.7.1
  //people_detector.setPersonClusterLimits (min_height, max_height,0.1,8.0);// for pcl 1.7.2
  
  people_detector.setMinimumDistanceBetweenHeads(heads_minimum_distance); 
  std::vector<Eigen::Vector3f> pp_center_list;  
  ros::Rate loop_rate(10);
  while (ros::ok() && (!viewer.wasStopped()))
  {
		if(new_cloud_available_flag)
		{
			//VectorXf ground_coeffs = getGroundCoeffs();
			//std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;
			new_cloud_available_flag = false;
      
      PointCloudT::Ptr cloud (new PointCloudT);
      pcl::copyPointCloud<PointT, PointT>(*cloud_obj, *cloud);
			
		    // Perform people detection on the new cloud:
		    std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
		    people_detector.setInputCloud(cloud);
		    people_detector.setGround(ground_coeffs);                    // set floor coefficients
		    people_detector.compute(clusters);                           // perform people detection

		    //ground_coeffs = people_detector.getGround();                 // get updated floor coefficients
       /* pcl::PointCloud<pcl::RGB>::Ptr rgb_image;
        std::vector< pcl::people::PersonCluster<PointT> > clusters;
        classifyperson(cloud,clusters,ground_coeffs,rgb_image);*/

        ground_coeffs = people_detector.getGround();
		    // Draw cloud and people bounding boxes in the viewer:
		    viewer.removeAllPointClouds();
		    viewer.removeAllShapes();
		    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
		    viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
		    unsigned int k = 0;
		    for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
		    {
          /*Eigen::Vector3f centroid = rgb_intrinsics_matrix * (it->getTCenter());
          centroid /= centroid(2);
          Eigen::Vector3f top = rgb_intrinsics_matrix * (it->getTTop());
          top /= top(2);
          Eigen::Vector3f bottom = rgb_intrinsics_matrix * (it->getTBottom());
          bottom /= bottom(2);
            
          //it->setPersonConfidence(person_classifier.evaluate(rgb_image, bottom, top, centroid, rgb_intrinsics_matrix, false));
          it->setPersonConfidence(person_classifier.evaluate(rgb_image, bottom, top, centroid, false));*/

          if(it->getPersonConfidence() > min_confidence) // draw only people with confidence above a threshold
		      {
		          // draw theoretical person bounding box in the PCL viewer:
              it->drawTBoundingBox(viewer, k);
		          k++;
              Eigen::Vector3f temp = it->getTCenter();
              if(temp(2) < detect_length)
              {
                pp_center_list.push_back(it->getTCenter());
              }
              
              std::cout << "Person " << k << " Position : X = " << temp(0) << " ,Y = " << temp(1) << " ,Z = " << temp(2) << std::endl;
              //myfile << temp(0) << "," << temp(1) << "," << temp(2)<< "|";
          }
        }
        //myfile << std::endl;
		    doMultiple_Tracking(pp_center_list,world_track_list,0.3);
        checktracklist(world_track_list);
        pp_center_list.clear();

        std::cout << "IN_TRACK -- " << world_track_list.size() << "persons" << endl;
        for(int i=0 ; i< world_track_list.size() ; i++)
        {
          Eigen::Vector3f tmpvect;
          tmpvect = world_track_list[i].points;
  
          std::cout<< "id : " << world_track_list[i].id <<  " framelostlifetime : " << world_track_list[i].framelostlifetime << std::endl
          << "x = " << tmpvect(0) << " y = " << tmpvect(1) << " z = " << tmpvect(2) << std::endl
          << "Track status = "<< world_track_list[i].istrack << " || framesage = " << world_track_list[i].framesage << "  frameincond = " << world_track_list[i].frameincond << std::endl;  
        }

        for(int i=0; i< world_track_list.size();i++)
        {
          Eigen::Vector3f out;
          out = world_track_list[i].points;
          std::string name = "sphere" + world_track_list[i].id;
          PointT pts;
          pts.x = out(0); pts.y = out(1); pts.z = out(2);
          viewer.removeShape(name.c_str());
          #ifdef COLOR_VISUALIZE
          viewer.addSphere (pts, 0.1, world_track_list[i].color(0), world_track_list[i].color(1), world_track_list[i].color(2), name.c_str());
          #else
          viewer.addSphere (pts, 0.1, 1, 0, 0.5, name.c_str());
          #endif
        }
         
        viewer.spinOnce();
        loop_rate.sleep();	
		}
  
	ros::spinOnce();
	}
  //myfile.close();
  return 0;
}
