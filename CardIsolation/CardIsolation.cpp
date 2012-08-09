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

using namespace std;
using namespace cv;


int thresh = 50;
IplImage* img = 0;
IplImage* img0 = 0;
CvMemStorage* storage = 0;
const char* wndname = "Input Image";
const char* croppedwndname = "Cropped Window";
const char* rotatedwnd = "Attempting to Rotate";
Mat* rotateDisplay;


///With any luck this should accept the square's sequence and the bounding box in rect
///and apply the affine transformation
Mat * cropRotate(CvRect *srcRect, CvSeq* srcPoints, IplImage *srcImg){

	///Create the matrix to store our points
	
		Mat * rotatedPtr = new Mat;
		Mat rotated = *rotatedPtr;

	if (srcRect->height > 0){
	cv::Mat src = srcImg;
	
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
	cv::Mat draw = src.clone();
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
	warpAffine(src, rotated, warpAffineMatrix, size);
	
//	cv::imwrite("rotated.jpg",rotated);
	}
	else
		fprintf(stderr, "ERROR: rotate didn't work");




//return rotated;
return rotatedPtr;
}

void  displayCropped(CvRect *srcRect, IplImage *cimg)
{
		if (srcRect->height > 0)
		{
		
		cout << endl << "X-Coord: " << srcRect->x; 
		cout << endl << "Y-Coord: " << srcRect->y;
		cout << endl << "Height " << srcRect->height;
		cout << endl << "Width" << srcRect->width;
		
		cout << endl << "About to start cropping" << endl;
		IplImage* cropped = cvCreateImage( cvSize(srcRect->width , srcRect->height), cimg->depth, cimg->nChannels );
		cvSetImageROI(cimg,*srcRect);
		//cvSetImageROI(img,cvRect(1,5,10,10));
		cvCopy(img,cropped,NULL);
		cvResetImageROI(cimg);
		cvNamedWindow( croppedwndname, 1 );
		cvShowImage( croppedwndname, cropped );
		}
		else
		cout << endl << "Warning: bad size" << endl;

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
CvSeq* findSquares4( IplImage* img, CvMemStorage* storage )
{
    CvSeq* contours;
    //int i, c, l, N = 11;
	int i, c, l, N = 2;
	CvSize sz = cvSize( img->width & -2, img->height & -2 );
    IplImage* timg = cvCloneImage( img ); // make a copy of input image
    IplImage* gray = cvCreateImage( sz, 8, 1 );
    IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
    IplImage* tgray;
    CvSeq* result;

	///creating a new bounding rectangle to put around the found card
	CvRect* rect = new CvRect;
	CvRect* cropRect = new CvRect;


    double s, t;
    // create empty sequence that will contain points -
    // 4 points per square (the square's vertices)
    CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
	
    // select the maximum ROI in the image
    // with the width and height divisible by 2
    cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));

    // down-scale and upscale the image to filter out the noise
    cvPyrDown( timg, pyr, 7 );
    cvPyrUp( pyr, timg, 7 );
    tgray = cvCreateImage( sz, 8, 1 );

    // find squares in every color plane of the image
    for( c = 0; c < 3; c++ )
    {
        // extract the c-th color plane
        cvSetImageCOI( timg, c+1 );
        cvCopy( timg, tgray, 0 );

        // try several threshold levels
        for( l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cvCanny( tgray, gray, 0, thresh, 5 );
                // dilate canny output to remove potential
                // holes between edge segments
                cvDilate( gray, gray, 0, 1 );
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );
            }

            // find contours and store them all as a list
            cvFindContours( gray, storage, &contours, sizeof(CvContour),
                CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

            // test each contour
           while( contours )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                result = cvApproxPoly( contours, sizeof(CvContour), storage,
                    CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
				//int largestContourAreaFound = 0;
				
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
				
                contours = contours->h_next;
            }
			////End the contour finding. 
			//rect.x = CvPoint(cvGetSeqElem(squares,0));
        }

    }


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



    // release all the temporary images
    cvReleaseImage( &gray );
    cvReleaseImage( &pyr );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );

    return squares;
}


// the function draws all the squares in the image
void drawSquares( IplImage* img, CvSeq* squares )
{
    CvSeqReader reader;
    IplImage* cpy = cvCloneImage( img );
    int i;

    // initialize reader of the sequence
    cvStartReadSeq( squares, &reader, 0 );

    // read 4 sequence elements at a time (all vertices of a square)
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
}




char* names[] = { "photo 2.JPG", "photo 3b.JPG", "photo 4b.JPG" , 0};



int main(int argc, char** argv)
{

//////////Webcam section
	CvCapture* capture = 0;
		
	IplImage* cam_curr_frame = 0; // current video frame
	IplImage* cam_gray_frame = 0; // grayscale version of current frame
	int camw, camh; // video frame size
	IplImage* cam_eig_image = 0;
	IplImage* cam_temp_image = 0;

		// Capture from a webcam
	capture = cvCaptureFromCAM(CV_CAP_ANY);
	//capture = cvCaptureFromCAM(0); // capture from video device #0
	if ( !capture) {
		fprintf(stderr, "ERROR: capture is NULL... Exiting\n");
		//getchar();
		return -1;
	}


		cvNamedWindow("CamWindow", 0); // allow the window to be resized
		while (true) {
		
		// Get one frame
		cam_curr_frame = cvQueryFrame(capture);
		if ( !cam_curr_frame) {
			fprintf(stderr, "ERROR: frame is null... Exiting\n");
			//getchar();
			break;
		}
		cvShowImage("CamWindow", cam_curr_frame);

		// If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
		// remove higher bits using AND operator
		//if ( (cvWaitKey(10) & 255) == 27)
		//	break;
		if ( (cvWaitKey(10) & 255) == 78)
		{
			img0 = cam_curr_frame;
			break;
		}



	}

	/////EndWebcam
	
	namedWindow("Rotated Window",CV_WINDOW_AUTOSIZE);

    int i, c;
    // create memory storage that will contain all the dynamic data
    storage = cvCreateMemStorage(0);
/*
    for( i = 0; names[i] != 0; i++ )
    {
        // load i-th image
        img0 = cvLoadImage( names[i], 1 );
        if( !img0 )
        {
            printf("Couldn't load %s\n", names[i] );
            continue;
        }

		*/

        //img = cvCloneImage( img0 );
		img = cvCreateImage(cvSize(640,480),img0->depth,img0->nChannels);
		cvResize(img0,img);
        // create window and a trackbar (slider) with parent "image" and set callback
        // (the slider regulates upper threshold, passed to Canny edge detector)
        cvNamedWindow( wndname, 1 );

        // find and draw the squares
        drawSquares( img, findSquares4( img, storage ) );

        // wait for key.
        // Also the function cvWaitKey takes care of event processing
        c = cvWaitKey(0);
        // release both images
        cvReleaseImage( &img );
        cvReleaseImage( &img0 );
        // clear memory storage - reset free space position
        cvClearMemStorage( storage );
    //    if( (char)c == 27 )
      //      break;
    
	cvDestroyWindow(croppedwndname);
    cvDestroyWindow( wndname );


	/////Webcam related

	cvReleaseCapture( &capture);
	cvDestroyWindow("CamWindow");
//	cvDestroyWindow(CORNER_EIG);


	return 0;
}

