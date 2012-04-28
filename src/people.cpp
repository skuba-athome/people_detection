#include "people.h"


void kinectCallBack(const sensor_msgs::ImageConstPtr& msg)
{
  	convertmsg2img(msg);
  	cvEqualizeHist(img,img);
	haveFace = 0;
  	//detect_face("cup.xml");
  	detect_face();
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
	  ros::Subscriber sub = n.subscribe("/camera/rgb/image_mono", 1, kinectCallBack);
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
IplImage * detect_people(){
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

void learn()
{
        int i, offset;

        // load training data
        nTrainFaces = loadFaceImgArray("data/train.txt");
        printf("%d\n",nTrainFaces);
        if( nTrainFaces < 2 )
        {
                fprintf(stderr,
                        "Need 2 or more training faces\n"
                        "Input file contains only %d\n", nTrainFaces);
                return;
        }
        doPCA();
        projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
        offset = projectedTrainFaceMat->step / sizeof(float);
        for(i=0; i<nTrainFaces; i++)
        {
                //int offset = i * nEigens;
                cvEigenDecomposite(
                        faceImgArr[i],
                        nEigens,
                        eigenVectArr,
                        0, 0,
                        pAvgTrainImg,
                        projectedTrainFaceMat->data.fl + i*offset);


        }
    storeTrainingData();
	if(is_init)
	{
    	system("espeak --stdout \'now i remember you\' | aplay");
		nNames++;
	}
	printf("debug segment.. \n");
}

void recognize()
{
  int i, nTestFaces  = 0;         // the number of test images
  CvMat * trainPersonNumMat = 0;  // the person numbers during training
  float * projectedTestFace = 0;
  if(!haveFace)
  {
		 return ;
  }
  g_count++;
  // load the saved training data
  if( !loadTrainingData( &trainPersonNumMat ) ) return;
  // project the test images onto the PCA subspace

  projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

          int iNearest, nearest, truth;

          // project the test image onto the PCA subspace
          cvEigenDecomposite(
                  faceImg,
                  nEigens,
                  eigenVectArr,
                  0, 0,
                  pAvgTrainImg,
                  projectedTestFace);

          iNearest = findNearestNeighbor(projectedTestFace);
          truth    = personNumTruthMat->data.i[i];
          nearest  = trainPersonNumMat->data.i[iNearest];

	printf("nearest = %d \n", nearest);
	g_nearest[nearest]++;
	if(g_count == nTestfaces){
		int index_max=0;
		int max = -1;
		for(int i = 0 ; i< nTestfaces ; i++)
		{
			if(g_nearest[i] > max)
			{
				max = g_nearest[i];
				index_max = i;
			}
		}
		char name[100];
		int num_tmp;
		FILE *fp = fopen("data/names.txt", "r");
		for(int fi=0;fi<=index_max;fi++)
			fscanf(fp, "%d %s", &num_tmp , name);	// read number of objects
		fclose(fp);
		printf("your name is name : %s\n",name);
		char cmd[1024];
		sprintf(cmd,  "espeak --stdout -s 150 \' your name is %s \' | aplay", name);
		system(cmd);
		is_recog = 0;
		g_count = 0;
		for(int i =0;i<nTestfaces;i++)	g_nearest[i]=0;
	}
}
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
        CvFileStorage * fileStorage;
        int i;

        // create a file-storage interface
        fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
        if( !fileStorage )
        {
                fprintf(stderr, "Can't open facedata.xml\n");
                return 0;
        }

        nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
        nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
        *pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
        eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
        projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
        pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
        eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
        for(i=0; i<nEigens; i++)
        {
                char varname[200];
                sprintf( varname, "eigenVect_%d", i );
                eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
        }

        // release the file-storage interface
        cvReleaseFileStorage( &fileStorage );

        return 1;
}
void storeTrainingData()
{
        CvFileStorage * fileStorage;
        int i;

        // create a file-storage interface
        fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

        // store all the data
        cvWriteInt( fileStorage, "nEigens", nEigens );
        cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
        cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
        cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
        cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
        cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
        for(i=0; i<nEigens; i++)
        {
                char varname[200];
                sprintf( varname, "eigenVect_%d", i );
                cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
        }
		printf("debug saveStorage\n");
        // release the file-storage interface
        cvReleaseFileStorage( &fileStorage );
}
int findNearestNeighbor(float * projectedTestFace)
{
        //double leastDistSq = 1e12;
        double leastDistSq = DBL_MAX;
        int i, iTrain, iNearest = 0;

        for(iTrain=0; iTrain<nTrainFaces; iTrain++)
        {
                double distSq=0;

                for(i=0; i<nEigens; i++)
                {
                        float d_i =
                                projectedTestFace[i] -
                                projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
                        distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis
                        //distSq += d_i*d_i; // Euclidean
                }

                if(distSq < leastDistSq)
                {
                        leastDistSq = distSq;
                        iNearest = iTrain;
                }
        }
        return iNearest;
}
void doPCA()
{
        int i;
        CvTermCriteria calcLimit;
        CvSize faceImgSize;

        IplImage* tmp_img = cvCreateImage( cvSize(100,100),IPL_DEPTH_8U, 1 );

        // set the number of eigenvalues to use
        nEigens = nTrainFaces-1;

        // allocate the eigenvector images
        faceImgSize.width  = faceImgArr[0]->width;
        faceImgSize.height = faceImgArr[0]->height;
        eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
        for(i=0; i<nEigens; i++)
        {
                eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
        }

        // allocate the eigenvalue array
        eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

        // allocate the averaged image
        pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

        // set the PCA termination criterion
        calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

        // compute average image, eigenvalues, and eigenvectors
        cvCalcEigenObjects(
                nTrainFaces,
                (void*)faceImgArr,
                (void*)eigenVectArr,
                CV_EIGOBJ_NO_CALLBACK,
                0,
                0,
                &calcLimit,
                pAvgTrainImg,
                eigenValMat->data.fl);

        cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);


}
int loadFaceImgArray(char * filename)
{
        FILE * imgListFile = 0;
        char imgFilename[512];
        int iFace, nFaces=0;


        // open the input file
        if( !(imgListFile = fopen(filename, "r")) )
        {
                fprintf(stderr, "Can\'t open file %s\n", filename);
                return 0;
        }

        // count the number of faces
        while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
        rewind(imgListFile);
        // allocate the face-image array and person number matrix
        faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
        personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );
        // store the face images in an array
        for(iFace=0; iFace<nFaces; iFace++)
        {
                // read person number and name of image file
                fscanf(imgListFile,
                       "%d %s", personNumTruthMat->data.i+iFace, imgFilename);
                faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

                if( !faceImgArr[iFace] )
                {
                        fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
                        return 0;
                }
        }

        fclose(imgListFile);
        return nFaces;
}

void saveFloatImage(const char *filename, const IplImage *srcImg)
{
	//cout << "Saving Float Image '" << filename << "' (" << srcImg->width << "," << srcImg->height << "). " << endl;
	IplImage *byteImg = convertFloatImageToUcharImage(srcImg);
	cvSaveImage(filename, byteImg);
	//cvReleaseImage(&byteImg);
	printf("debug_saveFloatImage\n");
}

IplImage* convertFloatImageToUcharImage(const IplImage *srcImg)
{
	IplImage *dstImg = 0;
	if ((srcImg) && (srcImg->width > 0 && srcImg->height > 0)) {

		// Spread the 32bit floating point pixels to fit within 8bit pixel range.
		double minVal, maxVal;
		cvMinMaxLoc(srcImg, &minVal, &maxVal);

		//cout << "FloatImage:(minV=" << minVal << ", maxV=" << maxVal << ")." << endl;

		// Deal with NaN and extreme values, since the DFT seems to give some NaN results.
		if (cvIsNaN(minVal) || minVal < -1e30)
			minVal = -1e30;
		if (cvIsNaN(maxVal) || maxVal > 1e30)
			maxVal = 1e30;
		if (maxVal-minVal == 0.0f)
			maxVal = minVal + 0.001;	// remove potential divide by zero errors.

		// Convert the format
		dstImg = cvCreateImage(cvSize(srcImg->width, srcImg->height), 8, 1);
		cvConvertScale(srcImg, dstImg, 255.0 / (maxVal - minVal), - minVal * 255.0 / (maxVal-minVal));
	}
	printf("debug convertFloatImage\n");
	return dstImg;
}

