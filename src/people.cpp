#include "people.h"


void kinectCallBack(const sensor_msgs::ImageConstPtr& msg)
{
  	convertmsg2img(msg);
  	cvEqualizeHist(img,img);
	haveFace = 0;
  	//detect_face("cup.xml");
  	detect_face("/home/xcerx/skuba_athome_main/people/haarcascade_frontalface_alt.xml");
  	if(is_recog){
	    recognize_realtime();
	}
  	cvShowImage("test",img);
  	cv::waitKey(10);
}

int main()
{
	  ros::init(argc,argv,"faces");
	  ros::NodeHandle n;
	  ros::NodeHandle nh("~");
	  nh.param("min_range", min_range_, 0.5);
	  nh.param("max_range", max_range_, 5.5);
	  ros::Subscriber sub = n.subscribe("/camera/rgb/image_color", 1, kinectCallBack);
	  ros::Subscriber sub2 = n.subscribe(TOPIC_CONTROL, 1, controlCallBack);
	  ros::Subscriber subDepth = n.subscribe("/camera/depth/image",1,depthCb);

	  ros::spin();
}

void convertmsg2img(const sensor_msgs::ImageConstPtr& msg)
{
	for(int i=0;i<1280*1024;i++)
		{
			img->imageData[i] = msg->data[i];
    }
	cvCvtColor ( imgRGB , img , CV_RGB2GRAY );
}

//===================================================================================================================
IplImage * detect_face(char filename[]){

  	CvMemStorage *storage = cvCreateMemStorage( 0 );
  	CvHaarClassifierCascade *cascade  = ( CvHaarClassifierCascade* )cvLoad( filename ,0 , 0, 0 );
  	CvRect *r = 0;

  	if(cascade == NULL)
  	{
  		printf("can't open haarcascade file . \n");
		return 0;
  	}

  	CvSeq* faces = cvHaarDetectObjects( img
										, cascade
										, storage
										, 1.1
										, 2
										, CV_HAAR_DO_CANNY_PRUNING
										, cvSize(65,65)  // ขนาด matrix ที่ใช้เริ่มในการหาใบหน้า
										);

  	float f_min = 2.0f;
  	for ( int i=0;i<( faces ? faces->total:0);i++)
  	{
          CvRect* tmp = (CvRect*)cvGetSeqElem(faces,i);
          if( dist[tmp->y+tmp->height/2][tmp->x+tmp->width/2] < f_min )
          {
		  	haveFace = 1;
            r = tmp;
			if( !isSkin(r) )  {
				r=0;
				haveFace = 0;
				continue;
			}
			f_min = dist[tmp->y+tmp->height/2][tmp->x+tmp->width/2];
          }
  	}

  	if(r == NULL ) // check for can't find
  	{
		return 0;
	}

	else
  	cvRectangle(img,cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),cvScalarAll(0.5),5,2,0);

	faceImg = cropImage(img, *r);
  	faceImg = resizeImage(faceImg,100,100);
  	cvEqualizeHist(faceImg,faceImg);

  	if(chkSave)
  	{
  	  	FILE * imgListFile;
    	// open the input file
    	while( !(imgListFile = fopen("./data/train.txt", "a+")) )
    	{
			printf("CAN'T OPEN train.txt , Create new file\n");
			system("touch /data/train.txt");
    	}
    	fprintf(imgListFile,"%d data/%s_%d.pgm\n",nNames,name,faceCount);
    	fclose(imgListFile);

    	char cstr[100];
    	sprintf(cstr, "./data/%s_%d.pgm",name , faceCount++);
    	printf(" the current face of '%s' into image '%s'.\n", name , cstr);
    	cvSaveImage(cstr, faceImg);
    	if(faceCount == MAX_FACES)
    	{
    	  	chkSave = 0;
    	  	faceCount = 0;
			learn();
	    }
 	 }

  	if(storage) cvReleaseMemStorage(&storage);
  	if(cascade) cvReleaseHaarClassifierCascade(&cascade);
  	return faceImg;
}
