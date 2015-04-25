#include <ros/ros.h>
#include <ros/package.h>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
//#include "athome_msgs/msg/navigation_goal.msg"
#include <athome_msgs/navigation_goal.h>
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
#include <fstream>

//typedef pcl::PointXYZRGBA PointT;
//typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using namespace Eigen;
using namespace std;

#define FRAME_OUT_CONDITION 12
#define FRAME_IN_CONDITION 3
#define FRAME_ENTRY_TRACKLIST 5
/*typedef struct{
  Eigen::Vector3f points;
  int id;
  int framescount;
  int in_frameprocess;
  int in_frametrack;
  bool istrack;
}person;*/

typedef struct{
  Eigen::Vector3f points;
  int id;
  int framesage;
  int frameincond;
  int framelostlifetime;
  bool istrack;
}person;

//std::vector<people_detection::PersonObject> world_track_list;
std::vector<person> world_track_list;


int lastavailable_id = 0;

float computenorm(Eigen::Vector3f A,Eigen::Vector3f B)
{
  float delx2 = (A(0)-B(0))*(A(0)-B(0));
  float dely2 = (A(1)-B(1))*(A(1)-B(1));
  float delz2 = (A(2)-B(2))*(A(2)-B(2));   
  return sqrt(delx2+dely2+delz2);
}

/*bool doMultiple_Tracking(std::vector<Eigen::Vector3f> &pp_newcenter_list,std::vector<person> &world, float disTH)
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
         float min = computenorm(world_temp[0].points,pp_newcenter_list[0]);
        int   index[] = {0,0};
        //cout << "size world = " << world_temp.size() << endl << "size center = " << pp_newcenter_list.size() <<endl;
        
        for(int i=0; i< world_temp.size() ; i++)
        {
          for(int j=0; j < pp_newcenter_list.size();j++ )
          {
           // nn_matching_table(i,j) = ( world_temp[i].points - pp_newcenter_list[j] ).squaredNorm();
            nn_matching_table(i,j) = computenorm(world_temp[i].points,pp_newcenter_list[j]);
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
              world[k].framescount = 0;//Updated Tracked points
              if(world[k].istrack == false)
              {
                world[k].in_frametrack++;
              }
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
          temp.framescount = 0;
          temp.in_frameprocess = 0;
          temp.in_frametrack = 0;
          #ifdef COLOR_VISUALIZE
          temp.color = generateTrackerColor();
          #endif
          temp.istrack = false;
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
              if(world[j].id == world_temp[i].id) ++world[j].framescount;
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
        temp.framescount = 0;
        temp.in_frameprocess = 0;
        temp.in_frametrack = 0;
        temp.istrack = false;

        #ifdef COLOR_VISUALIZE
        temp.color = generateTrackerColor();
        #endif
        
        world.push_back(temp);
      }
    }
}


void checktracklist(std::vector<person> &tracklist,int frameoutcond,int frameincond)
{
  //Remove or movein person
  for(int i=0 ; i < tracklist.size() ; i++ )
  { 
    //Remove Person
    if(frameoutcond <= tracklist[i].framescount)
    {
      tracklist.erase(tracklist.begin() + i);
    }
    //New Person checkframe
    if(tracklist[i].istrack == false)
    {
      //Lost Track of new entry frame
      if(tracklist[i].in_frameprocess != tracklist[i].in_frametrack)
      {
        tracklist.erase(tracklist.begin() + i);
      }
      else
      {
        tracklist[i].in_frameprocess++;
      }
      //Still track for new comming frame
      if(tracklist[i].in_frametrack > frameincond)
      {
        tracklist[i].istrack = true;
        tracklist[i].in_frameprocess = 0;
        tracklist[i].in_frametrack = 0;
      }
    }
  }
}*/
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
         float min = computenorm(world_temp[0].points,pp_newcenter_list[0]);
        int   index[] = {0,0};
        //cout << "size world = " << world_temp.size() << endl << "size center = " << pp_newcenter_list.size() <<endl;
        
        for(int i=0; i< world_temp.size() ; i++)
        {
          for(int j=0; j < pp_newcenter_list.size();j++ )
          {
           // nn_matching_table(i,j) = ( world_temp[i].points - pp_newcenter_list[j] ).squaredNorm();
            nn_matching_table(i,j) = computenorm(world_temp[i].points,pp_newcenter_list[j]);
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

int main(int argc, char **argv)
{
  ros::init(argc, argv, "people_detection");
  ros::NodeHandle n;
  

  std::string infilename = ros::package::getPath("people_detection") + "/sandbox/data2people1.txt";
  ifstream infile;
  infile.open(infilename.c_str());
  int line = 1;

  ros::Rate loop_rate(5);
  std::vector<Eigen::Vector3f> pp_center_list;
  while(ros::ok())
  { 
    if(infile)
    {
        string s1;
        
        if (!getline( infile, s1 )) break;
        
        stringstream ss( s1 );
        vector <string> record;
        vector<string> point3;

        while (ss)
        {
              string s;
              if (!getline( ss, s, '|' )) break;
              record.push_back( (s) );
        }

        
        for(int i=0 ; i < record.size() ; i++)
        {
          
          stringstream sss(record[i]);
          
          while(sss)
          {
            string s2;
            getline( sss, s2, ',' );
            if(!s2.empty())
            {
              point3.push_back(s2);
            }
              
          }
          
          if(!point3.empty())
          {
            Eigen::Vector3f vect3;
            vect3(0) = atof(point3[0].c_str());
            vect3(1) = atof(point3[1].c_str());
            vect3(2) = atof(point3[2].c_str());
          
            pp_center_list.push_back(vect3);
            point3.clear();
          }
          
        }

        /*Do Tracking*/
        cout << "--------------------------Current Line: " <<  line << " ,Pts amount: " <<  pp_center_list.size() <<std::endl;  
        line++;
        doMultiple_Tracking(pp_center_list,world_track_list,0.3);
        checktracklist(world_track_list);
        pp_center_list.clear();
        /*Show Result*/
        
        for(int i=0 ; i< world_track_list.size() ; i++)
        {
          Eigen::Vector3f tmpvect;
          tmpvect = world_track_list[i].points;
          /*std::cout<< "id : " << world_track_list[i].id <<  " framescount : " << world_track_list[i].framescount << std::endl
          << "x = " << tmpvect(0) << " y = " << tmpvect(1) << " z = " << tmpvect(2) << std::endl
          << "Track status = "<< world_track_list[i].istrack << " || Process = " << world_track_list[i].in_frameprocess << "  Track = " << world_track_list[i].in_frametrack << std::endl; */ 
          std::cout<< "id : " << world_track_list[i].id <<  " framelostlifetime : " << world_track_list[i].framelostlifetime << std::endl
          << "x = " << tmpvect(0) << " y = " << tmpvect(1) << " z = " << tmpvect(2) << std::endl
          << "Track status = "<< world_track_list[i].istrack << " || framesage = " << world_track_list[i].framesage << "  frameincond = " << world_track_list[i].frameincond << std::endl;  
        }
    }

  

    ros::spinOnce();
    loop_rate.sleep();
  }

  
  

}