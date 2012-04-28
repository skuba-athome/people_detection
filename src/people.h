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

CvMemStorage *storage = cvCreateMemStorage( 0 );
CvHaarClassifierCascade *cascade  = ( CvHaarClassifierCascade* )cvLoad( "/home/skubu-athome/skuba_athome_main/people/haarcascade_frontalface_alt.xml" ,0 , 0, 0 );

IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces
