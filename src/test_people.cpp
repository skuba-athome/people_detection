/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * main_ground_based_people_detection_app.cpp
 * Created on: Nov 30, 2012
 * Author: Matteo Munaro
 *
 * Example file for performing people detection on a Kinect live stream.
 * As a first step, the ground is manually initialized, then people detection is performed with the GroundBasedPeopleDetectionApp class,
 * which implements the people detection algorithm described here:
 * M. Munaro, F. Basso and E. Menegatti,
 * Tracking people within groups with RGB-D data,
 * In Proceedings of the International Conference on Intelligent Robots and Systems (IROS) 2012, Vilamoura (Portugal), 2012.
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl/ros/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

pcl::visualization::PCLVisualizer viewer("PCL Viewer");

PointCloudT cloud_obj;
bool cloud_available_flag = false;
void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_in)
{
	pcl::fromROSMsg(*cloud_in,cloud_obj);
	cloud_available_flag = true;
	//ROS_INFO("cloud size %d",cloud_obj.points.size());
	//ROS_INFO("r:%d g:%d b:%d x:%.2f",cloud_obj.points[0].r,cloud_obj.points[0].g,cloud_obj.points[0].b,cloud_obj.points[3000].x);	 		
}


void imageCallback(const sensor_msgs::Image::ConstPtr& img_in)
{
	if(!cloud_available_flag)return;
	
	PointCloudT::Ptr cloud (new PointCloudT);
  pcl::copyPointCloud<PointT, PointT>(cloud_obj, *cloud);

	pcl::RGB rgb_point;
	int j = 0;
	for (int i = 0; i < img_in->data.size(); i+=3)
	{
		rgb_point.b = img_in->data[i];
		rgb_point.g = img_in->data[i+1];
		rgb_point.r = img_in->data[i+2];

		cloud->points[j].b = rgb_point.b;
		cloud->points[j].g = rgb_point.g;
		cloud->points[j].r = rgb_point.r;
		j++;
	}

	viewer.removeAllPointClouds();
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "test_cloud");
	viewer.setCameraPosition(0,0,-2,0,-1,0,0);
	viewer.spinOnce();

	ROS_INFO("r:%d g:%d b:%d, r:%d g:%d b:%d",cloud->points[0].r,cloud->points[0].g,cloud->points[0].b,img_in->data[2],img_in->data[1],img_in->data[0]);


	//ROS_INFO("r:%d g:%d b:%d",cloud->points[0].r,cloud->points[0].g,cloud->points[0].b);	 	
	//ROS_INFO("j = %d",j);
}

/*pcl::PointCloud<pcl::RGB>::Ptr rgb_image(new pcl::PointCloud<pcl::RGB>);
void imageCallback(const sensor_msgs::Image::ConstPtr& img_in)
{
	rgb_image->points.resize(img_in->height*img_in->width);
	rgb_image->width = img_in->width;
	rgb_image->height = img_in->height;

	pcl::RGB rgb_point;
	int j = 0;
	for (int i = 0; i < img_in->data.size(); i+=3)
	{
		rgb_point.b = img_in->data[i];
		rgb_point.g = img_in->data[i+1];
		rgb_point.r = img_in->data[i+2];
		rgb_image->points[j++] = rgb_point;		
	}
	ROS_INFO("j = %d",j);
}*/




int main (int argc, char** argv)
{
	ros::init(argc, argv, "test_people");
	ros::NodeHandle n;
	ros::Subscriber	image_sub = n.subscribe("/camera/rgb/image_color", 1, imageCallback);
	ros::Subscriber	cloud_sub = n.subscribe("/camera/depth_registered/points", 1, cloudCallback);
	ros::spin();

  return 0;
}

