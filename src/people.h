#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <std_msgs/String.h>
#include <string.h>
#include <cxcore.h>
#include <cvaux.h>
#include <stdlib.h>

/***************************************************************/
using namespace std;
using namespace cv;
using namespace cv_bridge;

double min_range_;
double max_range_;
float dist[1280][1024];
int canPrintDepth = 0; // บางทีค่า depth มันมาช้ากว่า RGB พอเฟรมแรกแมร่งก็พัง ><

IplImage* img = cvCreateImage( cvSize(1280,1024),IPL_DEPTH_8U, 1 );
