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

