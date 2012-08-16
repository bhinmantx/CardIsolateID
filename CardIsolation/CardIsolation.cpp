// CardIsolation.cpp : Defines the entry point for the console application.
//


//////Note for anyone reading this code right now: This is a mishmash of code based on the "squares.c" tutorial 
//////released with OpenCV
#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY
//#define VIDEO_WINDOW "Webcam"
///Swapping VIDEO_WINDOW for ""CamWindow""



#include "stdafx.h"
#include <cv.h>
#include "cvaux.h"
#include <highgui.h>
#include "cxcore.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "C:\OpenCV2.1\src\cv\_cvimgproc.h"

using namespace std;
using namespace cv;


int thresh = 50;
///mats were pointers. swapping to objects 

//Mat * img = 0;
//Mat * img0 = 0;







const char* camwndname = "Input Image Cam";
const char* wndname = "square"; ////Is this what I am using to show the squares
const char* croppedwndname = "Cropped Window";
const char* rotatedwnd = "Attempting to Rotate";

Mat* rotateDisplay;

///Function declaration
vector<Rect> convertContoursToSqr(vector<vector<Point>>&);


///With any luck this should accept the square's sequence and the bounding box in rect
///and apply the affine transformation
////Modifying to accept a mat* instead of IplImage*
//Mat * cropRotate(CvRect *srcRect, CvSeq* srcPoints, IplImage *srcImg){
Mat * cropRotate(CvRect *srcRect, CvSeq* srcPoints, Mat srcImg){
	///Create the matrix to store our points
	
		Mat * rotatedPtr = new Mat;
		Mat rotated = *rotatedPtr;

	if (srcRect->height > 0){
	cv::Mat * src = &srcImg;
	
	vector<cv::Point> pointsToFix;
	
	///We need to enter the points as upper left, upper right, lower left, lower right??
	///Apparently not, apparently it's upper left, upper right, lower right, lower left
/*
	pointsToFix.push_back(cvPoint(srcRect->x,srcRect->y));
	pointsToFix.push_back(cvPoint((srcRect->x + srcRect->width),srcRect->y));
	pointsToFix.push_back(cvPoint((srcRect->x + srcRect->width),(srcRect->y + srcRect->height)));
	pointsToFix.push_back(cvPoint(srcRect->x,(srcRect->y + srcRect->height)));
*/
/*
	pointsToFix.push_back(cvPoint(srcRect->x,srcRect->y));

	pointsToFix.push_back(cvPoint((srcRect->x + srcRect->width),srcRect->y));

	pointsToFix.push_back(cvPoint((srcRect->x + srcRect->width),(srcRect->y + srcRect->height)));

	pointsToFix.push_back(cvPoint(srcRect->x,(srcRect->y + srcRect->height)));

	cout<< endl  << "X: " << srcRect->x << " y: " << srcRect->y << " of first point" << endl;
*/

///Instead of the src rec, let's try using squares

	//we need a seq reader
CvSeqReader reader;

cvStartReadSeq( srcPoints, &reader, 0);


if (srcPoints->total > 0){
//      CvPoint pt[4], *rect = pt;
//        int count = 4;
	cv::Point temp;
        // read 4 vertices
        CV_READ_SEQ_ELEM( temp, reader );
		cout << endl << "Can we reach this point?" << temp.x << endl;
		pointsToFix.push_back(temp);
        CV_READ_SEQ_ELEM( temp, reader );
        pointsToFix.push_back(temp);
		CV_READ_SEQ_ELEM( temp, reader );
		pointsToFix.push_back(temp);
        CV_READ_SEQ_ELEM( temp, reader );
		pointsToFix.push_back(temp);
}
else
fprintf(stderr, "ERROR:srcPoints is empty Exiting\n");





	/////Following stack overflow advice and dropping an image to disk
	const cv::Point* npoint = &pointsToFix[0];
	int n = (int)pointsToFix.size();

	std::cout<< endl << "cloning";
	cv::Mat draw = src->clone();
	std::cout<< endl << "drawing";
	cv::polylines(draw, &npoint, &n, 1, true, CV_RGB(0,255,0), 3, CV_AA);
//	std::cout<< endl << "saving";
//	imwrite("draw1.jpg",draw);


//	cvNamedWindow(rotatedwnd, 1);
	
//	cv::imshow( rotatedwnd, draw );
	//CvPoint* srcVerts = new CvPoint[3];
	//CvPoint* rotatedVerts;
	//	cv::Point2d srcVerts[3];
	//	srcVerts[0] = cvPoint(srcPoints[0]);

////Assemble a rotated rectangle out of that info
///I think I already have this, it's the src rect

	std::cout<< endl << "About to create the minAreaRect";


	cv::RotatedRect box = cv::minAreaRect(cv::Mat(pointsToFix));

	std::cout<< endl << "Array";
	///An array of points?
	cv::Point2f pts[4];
	///Copying those points from box?
	std::cout<< endl << "copying to box (possibly to)";
	box.points(pts);
	
	cv::Point2f src_vertices[3];
	src_vertices[0] = pts[0];
	src_vertices[1] = pts[1];
	src_vertices[2] = pts[3];

	cv::Point2f dst_vertices[3];
	dst_vertices[0] = cvPoint(0,0);
	dst_vertices[1] = cvPoint(box.boundingRect().width-1,0);
	dst_vertices[2] = cvPoint(0,box.boundingRect().height-1);

		std::cout<< endl << "Get Affine Transform";
	cv::Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

		std::cout<< endl << "Rotating?" << endl;
	//cv::Mat rotated;   ///original rotated decl
	cv::Size size(box.boundingRect().width, box.boundingRect().height);
	warpAffine(*src, rotated, warpAffineMatrix, size);
	
//	cv::imwrite("rotated.jpg",rotated);
	}
	else
		fprintf(stderr, "ERROR: rotate didn't work");




//return rotated;
return rotatedPtr;
}

void  displayCropped(CvRect *srcRect, Mat cimg)
{
		if (srcRect->height > 0)
		{
		
		cout << endl << "X-Coord: " << srcRect->x; 
		cout << endl << "Y-Coord: " << srcRect->y;
		cout << endl << "Height " << srcRect->height;
		cout << endl << "Width" << srcRect->width;
		
		cout << endl << "About to start cropping" << endl;
		//IplImage* cropped = cvCreateImage( cvSize(srcRect->width , srcRect->height), cimg.depth(), cimg.channels() );
		
//		Originally cropped was an IplImage * but we want to stick to Mats now. Trying a Mat* 
///		Mat * cropped = cvCreateImage( cvSize(srcRect->width , srcRect->height), cimg.depth(), cimg.channels() );
		Mat * cropped = new Mat(cvSize(srcRect->width, srcRect->height), cimg.depth(), cimg.channels());

		//cvSetImageROI(cimg,*srcRect);
		//cvSetImageROI(img,cvRect(1,5,10,10));
		
//////Setting the ROI in C++ is a bit different. We would create a specific rectangle for the ROI but we have the srcRect
		////We create a new image for that roi
		cv::Mat roi_for_cropped;
		
		////we copy that rectangle sized area from in this case, cimg
		roi_for_cropped = cimg(*srcRect);

		////now we copy that new Mat to cropped (originally used  cvCopy
		//		cvCopy(roi_for_cropped,cropped,NULL);
		cimg.copyTo(roi_for_cropped);


		///I don't think we need to worry about resetting it because we used that temp Mat
		//cvResetImageROI(cimg);
		cvNamedWindow( croppedwndname, 1 );
		cvShowImage( croppedwndname, cropped );
		
		}
		else
		cout << endl << "Warning: bad size" << endl;
////Do we need some kind of deletion for stuff like roi_for_cropped? 

}




// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}



// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
///Should change this to a vector I think.

//CvSeq * findSquares4( Mat img1, CvMemStorage *storage )
vector<Rect> findSquares4( Mat img1, CvMemStorage *storage )
{

	//	cout << endl << "Contours";
	////So cv::Seq is not what we want? Still using a cvSeq of....unknown
	/// We may need to swap to a vector (?)
	//CvSeq* contours = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
	/////Instead we will use a vector per OpenCV.org example

	vector<vector<Point>> contours;

	//	cout << endl << "contours Created";


	int i, c, l, N=11;

	//	cv::Size sz = cvSize(img.cols() & -2, img.rows() & -2);

	Size sz = img1.size();



	//  CvSeq* contours;
	//int i, c, l, N = 11;
	//	int i, c, l, N = 2;


	//CvSize sz = cvSize( img.cols() & -2, img.rows() & -2 );


	cout << endl << "findSquares4: set up a bunch of temp images";

	/////Make a clone of the input image
	Mat timg = img1.clone();

	cout << endl << "Timg declared!";

	cvWaitKey(0);

	imshow(camwndname,img1);

	Mat * gray = new Mat(sz,1);
	cout << endl << "gray DECLARED";

	cvtColor(timg,*gray,CV_RGB2GRAY);
	cout << endl << "gray Created";
	imshow(camwndname,*gray);

	cvWaitKey(0);
	cout << endl << "Trying to display timg";

	imshow(camwndname,timg);

	cout << endl << "Tried to display timg";

	cvWaitKey(0);




	//	IplImage* gray = cvCreateImage( sz, 8, 1 );
	//Mat gray(sz.height, sz.width, 1);

	//	Mat pyr(Size(240, 240));
	//	Mat pyr(240,240,1);
	Mat pyr(sz.height/2,sz.width/2,1);

	cout << endl << "Pyr Created";
	//	IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
	//Mat pyr(sz.height/2,sz.width/2, 1);


	//	Mat tgray(sz.width,sz.height,1);
	//	IplImage* tgray;


	//cout << endl << "tgray Created";


	CvSeq* result;


	///creating a new bounding rectangle to put around the found card
	//CvRect* rect = new CvRect;
	//CvRect* cropRect = new CvRect;
	Rect * cropRect = new Rect();

	double s, t;
	// create empty sequence that will contain points -
	// 4 points per square (the square's vertices)

	//cv::Seq * squares = new Seq(storage,sizeof(CvSeq));
	// select the maximum ROI in the image
	// with the width and height divisible by 2
	//



	//cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));
	//	timg.adjustROI( cvRect(0,0,sz.width,sz.height),);
	//timg.adjustROI(0,0,

	//	Rect * rect = new Rect(cvPoint(0,0),sz);
	//	Rect * rect = new Rect(0,0,288,288);
	Rect * rect = new Rect(0,0,timg.cols,timg.rows);
	//	Rect * rect = new Rect(0,0,timg.cols,timg.rows);



	cout << endl << "Trying for subImg" ;
	cout << endl << "timg Rows= " << timg.rows <<  " Columns = " << timg.cols;
	// down-scale and upscale the image to filter out the noise

	cout << endl << "about to down";


	//pyrDown(*gray,pyr,Size(subImg->cols/2,subImg->rows/2));
	pyrDown(*gray,pyr,Size(gray->cols/2,gray->rows/2));

	cout << endl << "Trying to display subImg after PyrDown";

	imshow(camwndname,*gray);

	cout << endl << "Tried to display new gray";
	cvWaitKey(0);


	cout << endl << "about to up";
	pyrUp( pyr, *gray,Size(gray->cols,gray->rows) );

	cout << endl << "pyrs complete";


	cout << endl << "Trying to display gray'd in other window";

	imshow(camwndname,*gray);

	cout << endl << "Tried to display gray";

	cout << gray->channels() << " Channels " << endl;
	cvWaitKey(0);


	////Switching to a purely Canny based detection
	////holder image
	Mat canny_output;
	Canny(*gray, canny_output, 100, 200, 3);
//	threshold(*gray,canny_output,10,255,CV_THRESH_BINARY);

	cout << " find contours " << endl;
	imshow(camwndname,canny_output);
	cvWaitKey(0);




///This works
findContours(canny_output,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
///this is unknown

//	vector<Vec4i> hierarchy;
//	findContours(canny_output,contours, hierarchy, CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);

	cout << endl << "Did we find contours? ";
	cout  << contours.size() <<  "Found" << endl;
	cvWaitKey(0);

	////Store the found squares here
	vector<Rect>boundRect(contours.size());

		cout << endl << " converting " <<endl ;
		
		boundRect =	convertContoursToSqr(contours);

		
		///srcConts should be a collection of contours from the "Found Contours"


cout << "Did that crash?" << endl;
cvWaitKey(0);


	
cout << "Rects found: " << boundRect.size();
cvWaitKey(0);

	//while( contours )
		/*	   while(true)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
//                result = cvApproxPoly( contours, sizeof(CvContour), storage,
  //                  CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );

				// square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
				//int largestContourAreaFound = 0;
		//		vector< vector<Point>> results;

				
		//		approxPolyDP(contours ,results, 5



				////Adapting code, experimenting with a 
				////bouding rect
				*rect = cvBoundingRect(result);
				double largestFound =0;
			
				double cRatio = (rect->height+0.0)/rect->width;
				
                if( result->total == 4 &&
                    (cvContourArea(result,CV_WHOLE_SEQ,0) > 20000) && (cvContourArea(result,CV_WHOLE_SEQ,0) < 300000) &&
                    cvCheckContourConvexity(result) && (cRatio<=1.40 && cRatio>=1.34))
					//cvCheckContourConvexity(result))
                {
					////Locating the (hopefully largest) contour matching the ratio of the car
					if((cvContourArea(result,CV_WHOLE_SEQ,0) > largestFound))
						{
							largestFound = (cvContourArea(result,CV_WHOLE_SEQ,0));
							
							//////we don't want to bound this rect anymore
							*cropRect = cvBoundingRect(result); 
							
							////But we have to keep it for testing
							//*cropRect = result;
						}
					
					s = 0;
					for( i = 0; i < 5; i++ )
                    {
                        // find minimum angle between joint
                        // edges (maximum of cosine)
                        if( i >= 2 )
                        {
                            t = fabs(angle(
                            (CvPoint*)cvGetSeqElem( result, i ),
                            (CvPoint*)cvGetSeqElem( result, i-2 ),
                            (CvPoint*)cvGetSeqElem( result, i-1 )));
                            s = s > t ? s : t;
                        }
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( s < 0.3 )
					{
						if(squares->total > 0)
						for( i = 0; i < 4; i++ )
							cvSeqRemove(squares,0);

                        for( i = 0; i < 4; i++ )
							cvSeqPush( squares,(CvPoint*)cvGetSeqElem( result, i ));
					}

				}
				

                // take the next contour
				
//                contours = contours->h_next;
				
            }
			
			////End the contour finding. 
			//rect.x = CvPoint(cvGetSeqElem(squares,0));
        }
	}

    

/*
		/////This is where we want to make our own sequence
	/////Filtered. It would contain only the largest of the 
	/////contours in "squares"
	//CvSeq* foundCard = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

		displayCropped(cropRect, img);
		cout <<  endl << "Finished the display" << endl;
		//char b;
		//cin>>b;


cout << "here" << endl;		

cout << "middle" << endl;

rotateDisplay = cropRotate(cropRect,squares,img);
cout << "there" << endl;
imshow("Rotated Window",*rotateDisplay);

*/

/*    // release all the temporary images
    cvReleaseImage( &gray );
    cvReleaseImage( &pyr );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );
*/
   // return squares;
	return boundRect;
}


vector<Rect> convertContoursToSqr(vector<vector<Point>> &srcConts){

	///srcConts should be a collection of contours from the "Found Contours"
	vector<vector <Point>> contours_poly(srcConts.size());
	vector<Rect> boundRect(srcConts.size());
	cout << endl << "SrcCounts size: " << srcConts.size();
	cout << endl << "fndSquares size: " << boundRect.size();
	cout << endl << "contours_poly size: " << contours_poly.size();



	for (int p = 0; p < srcConts.size(); p++)
	{
	
		approxPolyDP(Mat(srcConts[p]),contours_poly[p],.03,true);
		boundRect[p]= boundingRect(Mat(contours_poly[p]));

		


	}





//	approxPolyDP(Mat(srcConts[0]),contours_poly[0],3,true);

	cout << endl << "Loop finished";
	cout << endl << "fndSquares size: " << boundRect.size();
return boundRect;
}




// the function draws all the squares in the image
//void drawSquares( IplImage img, CvSeq* squares )
void drawSquares( Mat img, vector<Rect> sq)
{


	for(size_t i = 0, i<sq.size(),i++)
	{
	
		const Point* p = &sq[i][0];
		int n = (int)sq[i].size();
		polylines(img,&,&n,1,true, Scalar(0,255,0), 3, CV_AA);
	}



	/*

	cout << endl << "DrawSquares: create the reader, and then clone the img" ;
    CvSeqReader reader;
    IplImage* cpy = cvCloneImage( &img );
    int i;

    // initialize reader of the sequence
    cvStartReadSeq( squares, &reader, 0 );

    // read 4 sequence elements at a time (all vertices of a square)
	cout << endl << "Read the sequence elements";
    for( i = 0; i < squares->total; i += 4 )
    {
        CvPoint pt[4], *rect = pt;
        int count = 4;

        // read 4 vertices
        CV_READ_SEQ_ELEM( pt[0], reader );
        CV_READ_SEQ_ELEM( pt[1], reader );
        CV_READ_SEQ_ELEM( pt[2], reader );
        CV_READ_SEQ_ELEM( pt[3], reader );

        // draw the square as a closed polyline
        cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 1, CV_AA, 0 );
    }

    // show the resultant image
    cvShowImage( wndname, cpy );
    cvReleaseImage( &cpy );

	*/
}






int main(int argc, char** argv)
{

//////////Webcam section

	VideoCapture capture(0);
	if(!capture.isOpened()) //Check status of capture
	{	fprintf(stderr, "ERROR: capture is NULL... Exiting\n");
		return 0;
	}

	//IplImage* cam_curr_frame = 0; // current video frame
	//IplImage* cam_gray_frame = 0; // grayscale version of current frame
	//int camw, camh; // video frame size
	//IplImage* cam_eig_image = 0;
	//IplImage* cam_temp_image = 0;
	
	//Mat * cam_curr_frame = 0;
	Mat  cam_curr_frame; ///Can't use pointers to Mat's? ///remove init to 0
	Mat * cam_gray_frame = 0; ///curr frame and grayscale ver
	int camw, camh; //frame size
	Mat * cam_eig_image = 0;
	Mat * cam_temp_image = 0;

	Mat * img;
	Mat * img0 = new Mat();




		cv::namedWindow(camwndname,1);
		cv::namedWindow(camwndname,1);

		while (true) {
		
		// Get one frame
		//cam_curr_frame = cvQueryFrame(capture);
		capture >> cam_curr_frame;
		if ( cam_curr_frame.rows == 0) {
			fprintf(stderr, "ERROR: frame is null... Exiting\n");
			//getchar();
			break;
		}
		
		imshow(camwndname,cam_curr_frame);
		imshow(wndname,cam_curr_frame);
		//imshow(camwndname, cam_curr_frame); ///Gives us a preview of what the camera is seeing. 
											///currently freezes upon the switch to ID process.


		// If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
		// remove higher bits using AND operator
		//if ( (cvWaitKey(10) & 255) == 27)
		//	break;
//		if((waitKey(10) & 255)==78)

		
		if ( (cvWaitKey(10) & 255) == 78)
		{
			cout << endl << "img0 is about to be cam_curr_frame";
			*img0 = cam_curr_frame;
			cout << endl << "img0 is now cam_curr_frame";
			break;
		}


	}

	/////EndWebcam
	////
int i, c;


CvMemStorage * storage = cvCreateMemStorage(0);
//storage = new MemStorage();
/*
	cout << endl << "Create some windows";
	namedWindow(rotatedwnd,CV_WINDOW_AUTOSIZE);

    
    // create memory storage that will contain all the dynamic data
    //storage = cvCreateMemStorage(0);

	
	cout << endl << "Create storage";
//	storage = cvCreateMemStorage(0);
	storage = new MemStorage(0);
//	storage = new MemStorage(0);

*/



	//////different func to clone

		
		//img = cvCreateImage(cvSize(640,480),img0.depth(),img0.channels());
		//img = new Mat(img0.clone());
		
		img = new Mat();
		*img =(img0->clone());

		cout << endl << "Create the source img from the frame data as a matrix";
		///And instead of using the cvResize we use resize()
				cout << endl << "Resize";
		//resize(img0, img, cvSize(640,480),0,0,INTER_CUBIC);

				/////we're going to modify it so findSquares returns
				////a vector of rects

		 vector<Rect> sq = findSquares4(*img,storage);		



        // find and draw the squares
		cout << endl << "Draw some squares, if you find them";
       // drawSquares( IplImage(img), findSquares4( img, storage ) );


		drawSquares(*img, sq);

        // wait for key.
        // Also the function cvWaitKey takes care of event processing
        c = cvWaitKey(0);
       
		
		////Do we need to release these specifically?
		// release both images
       // cvReleaseImage( &img );
        //cvReleaseImage( &img0 );

		img->release();
		img0->release();
        // clear memory storage - reset free space position
        //cvClearMemStorage( storage );


		delete storage;
		storage = 0;
		//    if( (char)c == 27 )
      //      break;
    
	//cvDestroyWindow(croppedwndname);

	capture.release();		
	
	/////Webcam related


	//cvReleaseCapture( &capture);
//	cvDestroyWindow(CORNER_EIG);


	return 0;
}

