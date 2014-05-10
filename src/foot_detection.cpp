#include <ros/ros.h>
#include <stdio.h>
#include <string.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/String.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/Pose2D.h>
unsigned int num=0;
//unsigned int count=0;
unsigned int k = 1;
bool filter_enable=false;
bool robot_move=false;
int gap_foot=0;
unsigned int check_repeat = 0;
float filter = 0.08; //0.1
unsigned int n_obj=0;
unsigned int laser_range_noise[1500];  //720
float laser_range_1[1500];  //720
float laser_range_tmp[1500];  //720
int temp[750];
float temp_distance=0;
unsigned int robot_move_frame=0;
unsigned int frame=0;
char curve='n';
char state='a';
bool init_foot=false;
bool bt_foot=false;
int n_foot=0;
int n_point_foot=0;
int confirm_wall=0;

ros::Publisher pub_scan_2;
ros::Publisher pub_wall_people;

void scanCallback(const sensor_msgs::LaserScanPtr &scan)
{	
	unsigned int ranges_size = scan->ranges.size();
    geometry_msgs::Pose2D wall_people ;//= new geometry_msgs::Pose2D();
//    wall_people=new Pose2D();
    
//	printf("ranges_size %d",ranges_size);  Check Ranges_size
//	unsigned int fin_clone = 0;
	laser_range_noise[0]=9999;
    temp[0]=0;
    n_foot=0;
    n_point_foot=0;
    confirm_wall=0;
   // init_foot=false;
//    temp_distance=0;

    if(ranges_size <=720)
    {
        ///////////////////////   Copy Scan     ///////////////////////////////////
    	
    	for(unsigned int n = 0; n < ranges_size;n++)
    	{
    		laser_range_1[n] = scan->ranges[n];
    	//	sum_laser_range_1=sum_laser_range_1+laser_range_1[n];
    	}

        /////////////////////////////////////////////////////////////////////////////
        

        //////  check foot group /////////////
        for(int n=30; n < ranges_size-30; n++)
        {
            
            if(init_foot==true && scan->ranges[n] -temp_distance > 3 && state=='b')
                {
                    gap_foot++;
                }

            if(n_point_foot > 5&&gap_foot==1)
                {
                    gap_foot=0;
                    n_foot++;
                //    printf("N_Foot=%d \n",n_foot);    
                }
            if(n_foot >=3) 
            {
                confirm_wall++;
               // printf("Comfirm_wall = %d",confirm_wall);
            }
           // else
           //     wall_people.data = "NONE";
           // if(confirm_wall>=10)
           //     wall_people.data = "DETECTED";

            state ='a';
            if(laser_range_1[n]>=0.20&&laser_range_1[n]<1.7)
            {
               // if(temp_distance==0)     // init foot
               // {
               //     temp_distance = scan->ranges[n];               
               //     init_foot=true;
               // }

               // if(init_foot==true && scan->ranges[n] == 0 && temp_distance != 0)
               // printf("Init Foot %d \n",init_foot);
            //    std::cout << " Init foot " << init_foot << std::endl;
           //     printf("scan->range = %f temp_distance = %f",scan->ranges[n],temp_distance);
           //     if(init_foot==true && scan->ranges[n] -temp_distance > 3 )
           //     {
           //         gap_foot++;
           //     }

           //     if(gap_foot > 10)
           //     {
           //         gap_foot=0;
           //         bt_foot=false;
           //     //    temp_distance=0;

           //     }

               // std::cout << "Gap foot:" << gap_foot << "State Curve ::  "<< curve << std::endl;
                if(n>335&&n<385)
                {
                wall_people.x = scan->ranges[n];
                }

                if(confirm_wall>=5)
                {    
                    std::cout << "Wall people" << wall_people.x << std::endl; 
                    pub_wall_people.publish(wall_people);
                }

                if(init_foot==true && scan->ranges[n] > temp_distance ) 
                {
                    if( scan->ranges[n] - temp_distance < 0.7)  //distance
                    {
                        temp_distance = scan->ranges[n];
                        scan->ranges[n]=0;
                        curve = 'd';
                        //##//std::cout << "Gap foot:" << gap_foot << "State Curve ::  "<< curve << std::endl;
                        n_point_foot++;
                        state ='b';  // detect foot
                    }
                }

               

                    
                if(init_foot==true && scan->ranges[n] < temp_distance)
                {
                    if( temp_distance - scan->ranges[n] < 0.7)
                    {
                        temp_distance = scan->ranges[n];
                        scan->ranges[n]=0;
                        curve = 'u';
                        //##//std::cout << "Gap foot:" << gap_foot << "State Curve ::  "<< curve << std::endl;
                        n_point_foot++;
                        state ='b';  // detect foot
                    } 
                }

                if(init_foot==true && scan->ranges[n] == temp_distance)
                {
                    temp_distance = scan->ranges[n];
                    scan->ranges[n]=0;
                    n_point_foot++;
                    state ='b';  // detect foot
                    if(n<ranges_size/2)
                    {
                        curve = 'd';
                        //##//std::cout << "Gap foot:" << gap_foot << "State Curve ::  "<< curve << std::endl;
                    }
                    else
                    {
                        curve = 'u';
                        //##//std::cout << "Gap foot:" << gap_foot << "State Curve ::  "<< curve << std::endl;
                    }
                }

               

               // scan->ranges[n]=0;
                n_obj++;

                if(temp[0]==0&&temp_distance==0)     // init foot
                {
                    temp_distance = scan->ranges[n];               
                    scan->ranges[n]=0;
                    init_foot=true;
                }

                temp[0]=n;

            }    
            //temp_distance = scan->ranges[n];
            //temp_distance = 1.2;   // Fix Last distance (hack)
        
        }
        ////////////////////////////////////////////////////////
        std::cout << "Num_obj:\n" << n_obj << std::endl;
       // printf("Num_obj = %d",n_obj);
        n_obj=0;



      
    	//if(!filter_enable)
    	//num_noise=0;
    
    	/*if(filter_enable)
    	{
    
    		if(fin_clone==1)
    		{
    		fin_clone=0;
    		//printf("Clone complete  ************************************************* \n");
    
    			//------------------------------- Filter -> Noise -----------------------------------------------
    
    			for(num = 0; num < ranges_size;num++)
    			{
    				if(laser_range_1[num]-laser_range_2[num]>filter || laser_range_2[num]-laser_range_1[num]>filter
    				||laser_range_1[num]-laser_range_3[num]>filter || laser_range_3[num]-laser_range_1[num]>filter
    				||laser_range_1[num]-laser_range_4[num]>filter || laser_range_4[num]-laser_range_1[num]>filter
    				||laser_range_1[num]-laser_range_5[num]>filter || laser_range_5[num]-laser_range_1[num]>filter
    				||laser_range_2[num]-laser_range_4[num]>filter || laser_range_4[num]-laser_range_2[num]>filter)  
    			    ;	
    			}
    			///-------------------------------------------------------------------------------------------------
    	
    		}  
    
    			
    	//	filter_enable=true;
    //
    //		for(unsigned int j=1;j<num_noise;j++)     ///Check repeat	
    //		{		
    //			scan->ranges[laser_range_noise[j]]=0.0;	
    	//		printf("laser_range_noise[%d] = %d \n",j,laser_range_noise[j]);
    		//	printf("num_noise = %d \n",num_noise);
    //		}
    	}
        */
  /*  	if(sum_laser_range_5>0)
    	{
    	//	printf("sum_laser_range_1 = %d \n",sum_laser_range_1);
    	//	printf("sum_laser_range_2 = %d \n",sum_laser_range_2);
    	//	printf("sum_laser_range_3 = %d \n",sum_laser_range_3);
    	//	printf("sum_laser_range_4 = %d \n",sum_laser_range_4);
    	//	printf("sum_laser_range_5 = %d \n",sum_laser_range_5);
    		sum_laser_range_1=0;
    		sum_laser_range_2=0;
    		sum_laser_range_3=0;
    		sum_laser_range_4=0;
    		sum_laser_range_5=0;
    	}
  */
    }
	pub_scan_2.publish(scan);  
//    pub_wall_people.publish(wall_people);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "filter_scan");
  ros::NodeHandle n;
  ros::Subscriber filter_scan = n.subscribe("/scan", 1, scanCallback);
  //ros::Subscriber filter_scan_enable = n.subscribe("/vel_odom",5, filter_scan_enableCallback);
  pub_scan_2 = n.advertise<sensor_msgs::LaserScan>("/scan_3", 50);
  pub_wall_people = n.advertise<geometry_msgs::Pose2D>("/scan/wall_people", 1);
  ros::spin();
  return 0;
} 
