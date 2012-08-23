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

double angle( Point* pt1, Point* pt2, Point* pt0 );

vector<Rect> convertContoursToSqr(vector<vector<Point>> &srcConts);
//vector<vector<Point> > findCards( Mat img1);

//void findCards( Mat img1, vector<Point>*lastCard);

////older version of findcards with the vector of vectors of points
void findCards( Mat img1, vector<vector<Point> >*lastCard);

void convertContoursToPts(vector<vector<Point> > &srcConts,vector<vector<Point> > * foundSquares);
//vector<vector<Point> >* convertContoursToPts(vector<vector<Point> > &srcConts,vector<vector<Point> > * foundSquares);

//vector<Point> rectFromPoints(<vector>Point coords);

vector<Point> polyFromPoints(vector<Point>& coordpoints);

//void isolateCard(vector<vector<Point> > &foundCards, vector<Point> * lastCard);

void isolateCard(vector<vector<Point> > &foundCards);

void drawCardLines(vector<vector<Point> > pts, Mat * img);



void isolateCard(vector<vector<Point> > &foundCards)
{
	/////We find the biggest area

	Rect * cropRect;
	int largestArea = 0;
	vector<vector<Point> > * largestCard = new vector< vector<Point> >;
	///We iterate through found cards.

	for(int i=0;i<foundCards.size();i++)
	{
		int w = foundCards[i][1].x - foundCards[i][0].x;
		int h = foundCards[i][3].y - foundCards[i][0].y;

		cropRect = &Rect(foundCards[0][0].x,foundCards[0][0].y,w,h);
		cout << endl << "Area of the rect I found is " << abs(cropRect->area());
		//Is this the largest polygon?
		if(abs(cropRect->area()) > largestArea)
		{
			largestArea = abs(cropRect->area());
			//largestCard->clear();
			//largestCard->push_back(foundCards[i][3]);
			//largestCard->push_back(foundCards[i][2]);
			//largestCard->push_back(foundCards[i][1]);
			//largestCard->push_back(foundCards[i][0]);
			
			largestCard->push_back(foundCards[i]);
		}
		//		delete *cropRect;
	}

	foundCards.clear();
	foundCards.push_back(largestCard[0][0]);
	cout << "FoundCards size: " << foundCards.size();


	return;
}


void drawCardLine(vector<vector<Point> > pts, Mat * img)
{
	////takes a vector of vectors to points.
	cout << endl << "I see type 2 " << pts.size() << " many likely SHAPES in this";

	for(int q = 0; q < pts.size(); q++)
	{
		///we move through the vectors, which SHOULD contain 4 points each. SHOULD
		cout << endl << "I see " << pts[q].size() << " points in " << q;
		////Probably fastest just to draw every line, assuming there are 4 points
		cv::line(*img,pts[q][0],pts[q][1],Scalar(255,0,0),4,8,0);
		cv::line(*img,pts[q][1],pts[q][2],Scalar(0,255,0),4,8,0);
		cv::line(*img,pts[q][2],pts[q][3],Scalar(0,0,255),4,8,0);
		cv::line(*img,pts[q][3],pts[q][0],Scalar(0,255,255),4,8,0);
	}
		imshow(wndname,*img);
}

 





////Determines which points are in what positions
////for the purpose of making a (from top left to top right to lower right to lower left)
////Rectangle. This means identifying the corresponding corners of the given points
///and using those to calculate the x,y,width,height of a Rect

vector<Point> polyFromPoints(vector<Point>& coordpoints)
{

	if(coordpoints.size() < 4){
		fprintf(stderr, "ERROR: not enough coordpoints... Exiting\n");
		return coordpoints;
	}
	else
	{
		cout << endl << "coordSize: " << coordpoints.size();

		cout << endl << "Beginning point analysis";
		/////Find point "P" which is our upper left point
		int P = 0;
		for(int i =0;i<4;i++)
		{
			////looking for point with lowest x,y vals
			cout << endl<< "CoordPoints[" << i << "]: y: " << coordpoints[i].y << "x: " << coordpoints[i].x;
			if (coordpoints[i].y < coordpoints[P].y)
				P = i;
			else if (coordpoints[P].y == coordpoints[i].y && coordpoints[P].x > coordpoints[i].x)
				P = i;
		}

		///We now have "P" the vector index of the upper left corner. 
		///Now we calculate which angles have the smallest angle between the vector 
		///created from P and P.x+1,P.y) and P to the comparison point. 
		cout << endl<< "Loop complete. Now dealing with angles" <<endl;
		vector<double> angles(4);

		for(int j=0;j<4;j++)
		{

			if(j != P)
			{

				angles[j] = abs(angle(&Point(coordpoints[P].x+1,coordpoints[P].y), &coordpoints[j], &coordpoints[P]));
				cout << endl<< "Angle: " << angles[j];
			}
			else angles[P] = -1; 

		}////finish making those angles


	}


	return coordpoints;
}


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
//double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
double angle( Point* pt1, Point* pt2, Point* pt0 )
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


	//	CvSeq* result;


	///creating a new bounding rectangle to put around the found card
	//CvRect* rect = new CvRect;
	//CvRect* cropRect = new CvRect;
	//Rect * cropRect = new Rect();

	double s, t;
	// create empty sequence that will contain points -
	// 4 points per square (the square's vertices)

	//cv::Seq * squares = new Seq(storage,sizeof(CvSeq));
	// select the maximum ROI in the image
	// with the width and height divisible by 2
	//




	// down-scale and upscale the image to filter out the noise


	pyrDown(*gray,pyr,Size(gray->cols/2,gray->rows/2));


	///Display the intervening images
	imshow(camwndname,*gray);

	pyrUp( pyr, *gray,Size(gray->cols,gray->rows) );

	imshow(camwndname,*gray);
	////Switching to a purely Canny based detection
	////holder image
	Mat canny_output;
	Canny(*gray, canny_output, 100, 200, 3);
	//	threshold(*gray,canny_output,10,255,CV_THRESH_BINARY);

	imshow(camwndname,canny_output);
	cvWaitKey(0);




	///This works
	findContours(canny_output,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
	///this is unknown
	////Store the found squares here
	//	vector<Rect>boundRect(contours.size());
	vector<Rect> boundRect;
	cout << endl << " converting " <<endl ;

	boundRect =	convertContoursToSqr(contours);


	///srcConts should be a collection of contours from the "Found Contours"


	cout << "Did that crash?" << endl;
	cvWaitKey(0);



	cout << "Rects found: " << boundRect.size();
	cvWaitKey(0);


	/*
	/////This is where we want to make our own sequence
	/////Filtered. It would contain only the largest of the 
	/////contours in "squares"
	//CvSeq* foundCard = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

	displayCropped(cropRect, img);
	cout <<  endl << "Finished the display" << endl;
	//char b;
	//cin>>b;
	*/
	return boundRect;
}


////This version returns the vector of points of (hopefully) the card. 
//vector<vector<Point> > findCards( Mat img1 )
////
void findCards( Mat img1, vector<vector<Point> > *cards )
{
cout << "void findCards( Mat img1, vector<Point> *cards )" << endl;

	//Store shapes we find
	vector<vector<Point>> contours;

	int i, c, l, N=11;
	Size sz = img1.size();
	/////Make a clone of the input image
	Mat timg = img1.clone();

	cout << endl << "Timg declared!";

	cvWaitKey(0);

	imshow(camwndname,img1);

	Mat * gray = new Mat(sz,1);
	cout << endl << "gray DECLARED";

	cvtColor(timg,*gray,CV_RGB2GRAY);
	cout << endl << "gray Created";
	///show gray scale
	imshow(camwndname,*gray);
	//waitKey(0);
	////Show current temp image (timg)
	imshow(camwndname,timg);


	///Pyr is a temp image to hold data for when we up/down scale the img
	Mat pyr(sz.height/2,sz.width/2,1);
	// down-scale and upscale the image to filter out the noise

	pyrDown(*gray,pyr,Size(gray->cols/2,gray->rows/2));
	///Display the intervening images

	pyrUp( pyr, *gray,Size(gray->cols,gray->rows) );

	imshow(camwndname,*gray);
	////Switching to a purely Canny based detection
	////holder image
	Mat canny_output;
	Canny(*gray, canny_output, 100, 200, 3);
	//	threshold(*gray,canny_output,10,255,CV_THRESH_BINARY);

	imshow(camwndname,canny_output);
	cvWaitKey(0);

	///This works
	findContours(canny_output,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);


	/////////////////////////////////////////////////////////////////////////
	//vector<vector<Point> >  *foundCards = new vector<vector<Point> >;
	//////////////////////////////////////////REDUNDANT

	cout << endl << " converting " <<endl ;

	//		boundRect =	convertContoursToSqr(contours);

////This function makes a vector of vectors of 4 points.
	convertContoursToPts(contours,cards);
	
	//convertContoursToPts(contours,lastCard);
////Now that we have a vector of cards, we should narrow it down to a single card. 

/////I don't think this is working, going to try removing it
	//cout << endl << "Found cards: " << cards->size();
	//isolateCard(*foundCards,lastCard);
	//	cout << endl << "Found cards after isolation: " << cards->size();
	///srcConts should be a collection of contours from the "Found Contours"


	cvWaitKey(0);


	/*
	/////This is where we want to make our own sequence
	/////Filtered. It would contain only the largest of the 
	/////contours in "squares"
	//CvSeq* foundCard = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

	displayCropped(cropRect, img);
	cout <<  endl << "Finished the display" << endl;
	//char b;
	//cin>>b;
	*/
	//	return *foundCards;
	return;
}




vector<Rect> convertContoursToSqr(vector<vector<Point>> &srcConts){

	///srcConts should be a collection of contours from the "Found Contours"
	vector<vector <Point>> contours_poly(srcConts.size());
	vector<Rect> boundRect;
	vector<Point> approx;

	cout << endl << "SrcCounts size: " << srcConts.size();
	cout << endl << "fndSquares size: " << boundRect.size();
	cout << endl << "contours_poly size: " << contours_poly.size();


	for (size_t p = 0; p < srcConts.size(); p++)
	{

		approxPolyDP(Mat(srcConts[p]), approx, arcLength(Mat(srcConts[p]), true)*0.02, true);

		if(approx.size() == 4 && fabs(contourArea(Mat(approx)))>1000 && isContourConvex(Mat(approx)))

		{
			double maxCosine = 0;

			for( int j = 2; j < 5; j++ )
			{
				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(angle(&approx[j%4], &approx[j-2], &approx[j-1]));
				cout << endl << "Cosine is " << cosine;
				maxCosine = MAX(maxCosine, cosine);
				cout << "J: " << j;
			}
			// if cosines of all angles are small
			// (all angles are ~90 degree) then write quandrange
			// vertices to resultant sequence
			if( maxCosine < 0.3 )
			{

				///the following works but not point to point
				//boundRect.push_back (boundingRect(Mat(srcConts[p])));

				///We don't want rects anymore. Eh they might be useful for ROI
				///But that's later.

				cout << endl << "X, Y of 0" << approx[0].x << " " << approx[0].y;
				cout << endl << "X, Y of 0" << approx[1].x << " " << approx[1].y;
				cout << endl << "X, Y of 0" << approx[2].x << " " << approx[2].y;
				cout << endl << "X, Y of 0" << approx[3].x << " " << approx[3].y;




				////Need to implement this to be safe
				////This is broken I think
				//polyFromPoints(approx);

			}
		}

	}

	cout << endl << "boundRect size: " << boundRect.size();

	return boundRect;
}



////This function should be pulling the contours that have 4 corners and pushing them on to the foundsquares???

//void convertContoursToPts(vector<vector<Point>> &srcConts, vector<vector<Point> > *foundSquares){
void convertContoursToPts(vector<vector<Point>> &srcConts, vector<vector<Point> > *foundSquares){
	//vector<vector<Point> >* convertContoursToPts(vector<vector<Point>> &srcConts, vector<vector<Point> > *foundSquares){
	///srcConts should be a collection of contours from the "Found Contours"
	vector<vector <Point>> contours_poly(srcConts.size());
	//	vector<Rect> boundRect;
	vector<Point> approx;

	cout << endl << "SrcCounts size: " << srcConts.size();
	//	cout << endl << "fndSquares size: " << boundRect.size();
	cout << endl << "contours_poly size: " << contours_poly.size();



	for (size_t p = 0; p < srcConts.size(); p++)
	{
		approxPolyDP(Mat(srcConts[p]), approx, arcLength(Mat(srcConts[p]), true)*0.02, true);

		if(approx.size() == 4 && fabs(contourArea(Mat(approx)))>1000 && isContourConvex(Mat(approx)))

		{
			double maxCosine = 0;

			for( int j = 2; j < 5; j++ )
			{
				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(angle(&approx[j%4], &approx[j-2], &approx[j-1]));
				cout << endl << "Cosine is " << cosine;
				maxCosine = MAX(maxCosine, cosine);
				cout << "J: " << j;
			}
			// if cosines of all angles are small
			// (all angles are ~90 degree) then write quandrange
			// vertices to resultant sequence
			if( maxCosine < 0.3 )
			{

				///the following works but not point to point
				//boundRect.push_back (boundingRect(Mat(srcConts[p])));

				///We don't want rects anymore. Eh they might be useful for ROI
				///But that's later.
				cout << endl << "X, Y of 0" << approx[0].x << " " << approx[0].y;
				cout << endl << "X, Y of 1" << approx[1].x << " " << approx[1].y;
				cout << endl << "X, Y of 2" << approx[2].x << " " << approx[2].y;
				cout << endl << "X, Y of 3" << approx[3].x << " " << approx[3].y;
				cout << endl << "Push it";
				foundSquares->push_back(approx);

				

				////Need to implement this to be safe
				cout << endl << "PolyfromPts";
				//								polyFromPoints(approx);

			}
		}

	}

	//	cout << endl << "boundRect size: " << boundRect.size();

	//return boundRect;
	return;
}




// the function draws all the squares in the image
//void drawSquares( IplImage img, CvSeq* squares )
void drawSquares( Mat img, vector<Rect> sq)
{
	int  scalarColorB = 0;
	for(size_t i = 0; i<sq.size();i++)
	{
		cout << endl << "I drew a square";
		/////////////////////////////////////////////
		Point pts[4];
		pts[0] = Point(sq[i].x,sq[i].y);
		pts[1] = Point(sq[i].x+sq[i].width,sq[i].y);
		pts[2] = Point(sq[i].x,sq[i].y+sq[i].height);
		pts[3] = Point(sq[i].x+sq[i].width,sq[i].y+sq[i].height);
		int n =4;

		line(img,pts[0],pts[1],Scalar(scalarColorB,255,0),4,8,0);

		line(img,pts[1],pts[3],Scalar(scalarColorB,255,0),4,8,0);

		line(img,pts[3],pts[2],Scalar(scalarColorB,255,0),4,8,0);

		line(img,pts[2],pts[0],Scalar(scalarColorB,255,0),4,8,0);
		cout << endl << "Scalar color: " << scalarColorB; 
		scalarColorB = 255 - scalarColorB;
		imshow(camwndname,img);
		waitKey(0);

		//polylines(img,&pts,

		//		int n = 4;
		//		Point pts[3];

		//		const Point* p = &sq[i][0];
		//	int n = (int)sq[i].size();
		//		polylines(img,&,&n,1,true, Scalar(0,255,0), 3, CV_AA);


		//////////////////////////
		//for( size_t i = 0; i < squares.size(); i++ )
		//  {
		//    const Point* p = &squares[i][0];
		//  int n = (int)squares[i].size();
		//polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);




	}



}






int main(int argc, char** argv)
{

	//////////Webcam section

	VideoCapture capture(0);
	if(!capture.isOpened()) //Check status of capture
	{	fprintf(stderr, "ERROR: capture is NULL... Exiting\n");
	return 0;
	}


	Mat  cam_curr_frame; ///Can't use pointers to Mat's? ///remove init to 0
	Mat * cam_gray_frame = 0; ///curr frame and grayscale ver
	int camw, camh; //frame size
	Mat * cam_eig_image = 0;
	Mat * cam_temp_image = 0;

	Mat * img;
	Mat * img0 = new Mat();

	///Create a window that shows input from cam
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

	///Make a copy of the image we pulled from the camera
	img = new Mat();
	*img =(img0->clone());

	vector<vector<Point> > * Cards = new vector<vector <Point> >;

	findCards(*img, Cards);
	cout << endl << "After findCards Cards size is " << Cards->size();
 

	isolateCard(*Cards);

	cout << endl << "Cards vector size = " << Cards->size();


	// find and draw the squares

	//cout << endl << "X/Y of last card " << lastCard[0]->x << " " << lastCard[0]->y;
	
	cout << endl << "Draw some lines, if you find them" << Cards->size();

//	drawCardLine(*lastCard, img);     
	drawCardLine(*Cards, img);

	// drawSquares( IplImage(img), findSquares4( img, storage ) );
	imshow(camwndname,*img);

	//drawSquares(*img, sq);

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
	//		storage = 0;
	//    if( (char)c == 27 )
	//      break;

	//cvDestroyWindow(croppedwndname);

	capture.release();		

	/////Webcam related


	//cvReleaseCapture( &capture);
	//	cvDestroyWindow(CORNER_EIG);


	return 0;
}

