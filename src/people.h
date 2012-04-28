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

//#define RGB
using namespace std;
using namespace cv;
using namespace cv_bridge;

#define MAX_FACES 10
#define TOPIC_CONTROL "/cmd_state"
#define nTestfaces 10

IplImage* imgRGB = cvCreateImage( cvSize(1280,1024),IPL_DEPTH_8U, 3 );
IplImage* img = cvCreateImage( cvSize(1280,1024),IPL_DEPTH_8U, 1 );
cv::Mat depthImg ;
cv_bridge::CvImagePtr bridge;
IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces
IplImage * faceImg;
int faceCount = 0;
int chkSave = 0; // check for can save !?
int nNames = 0;
char name[100];
double min_range_;
double max_range_;
float dist[1280][1024];
int canPrintDepth = 0; // บางทีค่า depth มันมาช้ากว่า RGB พอเฟรมแรกแมร่งก็พัง ><
int haveFace = 0;
int g_nearest[20];
int g_count = 0;
int is_recog = 0;
int is_init = 0;

void convertmsg2img(const sensor_msgs::ImageConstPtr& msg);
IplImage * detect_people();
//// Function prototypes
void learn();
void recognize();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();
IplImage* cropImage(const IplImage *img, const CvRect region);
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
void saveFloatImage(const char *filename, const IplImage *srcImg);
void recognize_realtime();

int isSkin(CvRect *r);
unsigned minRGB(unsigned char r,unsigned char g,unsigned b);
unsigned maxRGB(unsigned char r,unsigned char g,unsigned b);
