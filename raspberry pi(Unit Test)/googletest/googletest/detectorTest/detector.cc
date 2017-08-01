/*************************************************************************/
/**
* @file detector.cpp
* @brief detect area of tongue from image file

* Detect the area of tongue from image file

* @author RG Jong
* @date 2017/7/21 version 2.0
*/
/* Copyright(C) 2017 Jinhu All right reserved */
/**************************************************************************/

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>

#include "detector.h"

using namespace std;
using namespace cv;

// temporary vector arrays to be used in function
vector<Mat> g_tempChannels;
vector<vector<Point>> g_contours1;
vector<vector<Point>> g_contours2;
vector<vector<Point>> g_contours3;
vector<Point> g_tempContour1;
vector<Point> g_tempContour2;
vector<Point> g_tempContour3;
vector<Rect> g_boundingRects;


/**
* @brief differentiate image

* Calculate horizontal difference of give image

* @param src input image to be processed(must be gray image)
* @param dst output image of differentiation(gray image)
* @param step step of horizontal differentiation
* @author Pai Jin
* @date 2017/7/17 (demo version for testing algorithm)
*/
void differentiate(Mat& src, Mat& dst, int step, int thres) {
	int w, h;
	w = src.cols;
	h = src.rows;
	dst = Mat::zeros(Size(w, h), src.type());
	for (int i = 0; i < h; i++) {
		for (int j = step; j < w - step; j++) {
			if (src.at<uchar>(i, j + step) > 0 && src.at<uchar>(i, j - step) > 0) {
				int diff = abs(src.at<uchar>(i, j - step) - src.at<uchar>(i, j + step));
				dst.at<uchar>(i, j) = (diff > thres) ? 255 : 0;
			}
		}
	}
}


/**
* @brief Process image to detect edge of tongue

* Process image to detect edge of tongue

* @param filePath the path of image file to be processed
* @param bTest 0: normal mode(default), 1: unit test mode(should not allow any output)
* @return DT_SUCCESS: finished successfully, DT_FILENOTEXIST: file not found
* @author Pai Jin
* @date 2017/7/17 (demo version for testing algorithm)
*/
int tongueDetectionAlgorithm(const char* filePath, int bTest) {
	// read image file to be processed
	Mat matOrg = imread(filePath);
	if(matOrg.rows == 0){
		if(!bTest) printf("Can not open the file: %s\n", filePath);
		return DT_FILENOTEXIST;
	}
	Mat matResizedOrg;
	resize(matOrg, matResizedOrg, Size(matOrg.cols / 2, matOrg.rows / 2));
	if(!bTest) imshow("Original Image", matResizedOrg);

	// apply blur to remove noise
	GaussianBlur(matResizedOrg, matResizedOrg, Size(5, 5), 1);

	// get channels of hue, saturation, value
	Mat matOrgHSV;																		// HSV image
	cvtColor(matResizedOrg, matOrgHSV, CV_BGR2HSV);
	split(matOrgHSV, g_tempChannels);

	Mat matVPolor;																		// will be the result of polor coordination transform
	Point2f ptCenter = Point2f(matResizedOrg.cols / 2, matResizedOrg.rows / 2);			// center point of polor coordination transform
	double dRadius = (matResizedOrg.cols > matResizedOrg.rows) ? matResizedOrg.cols / 2 : matResizedOrg.rows / 2;	// radius of PCT
	// apply polor coordination transformation
	linearPolar(g_tempChannels[2], matVPolor, ptCenter, dRadius, INTER_LINEAR + WARP_FILL_OUTLIERS);

	resize(matVPolor, matVPolor, Size((int)dRadius / 2, matVPolor.rows));

	Mat matVDiff;
	// calculate differentiation image
	differentiate(matVPolor, matVDiff, 2, 2);

	Mat matElem = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1, 1));
	// erode image to remove unnecessary connections of contours
	erode(matVDiff, matVDiff, matElem);

	Mat matVDiffResv = matVDiff.clone();

	cvtColor(matVPolor, matVPolor, CV_GRAY2BGR);

	Mat matProcVPolor = Mat::zeros(Size(matVPolor.cols, matVPolor.rows), CV_8UC1);

	// find contours
	findContours(matVDiffResv, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	g_contours2.clear();
	for (int i = 0; i < g_contours1.size(); i++) {
		Rect rt = boundingRect(g_contours1[i]);
		if (rt.height > matVDiff.rows / 8) {

			approxPolyDP(g_contours1[i], g_tempContour1, 4, true);

			g_tempContour2.clear();
			for (int j = 0; j < g_tempContour1.size() - 1; j++) {
				int id1 = (j == 0) ? g_tempContour1.size() - 1 : j - 1;
				int id2 = j;
				int id3 = (j == g_tempContour1.size() - 1) ? 0 : j + 1;

				int x = (g_tempContour1[id1].x + g_tempContour1[id2].x + g_tempContour1[id3].x) / 3;
				int y = (g_tempContour1[id1].y + g_tempContour1[id2].y + g_tempContour1[id3].y) / 3;

				g_tempContour2.push_back(Point(x, y));
			}

			polylines(matVPolor, g_tempContour1, true, Scalar(0, 0, 255));

			g_contours2.push_back(g_tempContour2);
		}
	}

	int iMaxId = -1;
	int iMaxHeight = 0;
	for (int i = 0; i < g_contours2.size(); i++) {
		Rect rtBound = boundingRect(g_contours2[i]);
		if (rtBound.height > iMaxHeight) {
			iMaxHeight = rtBound.height;
			iMaxId = i;
		}
	}

	if (iMaxHeight < matProcVPolor.rows * 8 / 10) {
		for (int i = 0; i < g_contours2.size(); i++) {
			fillPoly(matProcVPolor, g_contours2, Scalar(255));
		}
	}
	else {
		g_contours3.clear();
		g_contours3.push_back(g_contours2[iMaxId]);
		fillPoly(matProcVPolor, g_contours3, Scalar(255));
	}

	int* iMarks = new int[matProcVPolor.rows];
	for (int i = 0; i < matProcVPolor.rows; i++) {
		iMarks[i] = -1;

		int iLPos = -1, iRPos = -1;
		for (int j = 0; j < matProcVPolor.cols; j++) {
			if (matProcVPolor.at<uchar>(i, j) == 255) {
				iRPos = j;
				if (iLPos == -1) iLPos = j;
			}
		}

		if (iLPos >= 0) {
			iMarks[i] = (iLPos + iRPos) / 2;
		}
	}

	g_contours3.clear();
	g_tempContour2.clear();
	// get edge line of tongue from contour in polor coordinate
	double m = 2 * 3.141592 / matProcVPolor.rows;
	for (int i = 0; i < matProcVPolor.rows; i++) {
		if (iMarks[i] >= 0) {
			double dAngle = m*i;
			double dR = dRadius * iMarks[i] / matProcVPolor.cols;
			int x = matResizedOrg.cols / 2 + (int)(cos(dAngle)*dR);
			int y = matResizedOrg.rows / 2 + (int)(sin(dAngle)*dR);
			if (i == 108) {
				int k = 0;
			}
			g_tempContour2.push_back(Point(x, y));
		}
	}

	delete[] iMarks;

	// smooth polyline
	g_tempContour3.clear();
	int iSmooth = 10;
	for (int i = 0; i < g_tempContour2.size(); i++) {

		int x = 0;
		int y = 0;
		for (int j = -iSmooth; j <= iSmooth; j++) {
			int id = (i + j + g_tempContour2.size()) % g_tempContour2.size();
			x += g_tempContour2[id].x;
			y += g_tempContour2[id].y;
		}

		x /= (iSmooth * 2 + 1);
		y /= (iSmooth * 2 + 1);

		g_tempContour3.push_back(Point(x, y));
	}

	g_contours3.push_back(g_tempContour3);

	// create mask image
	Mat matMask = Mat::zeros(matResizedOrg.rows, matResizedOrg.cols, CV_8UC1);
	fillPoly(matMask, g_contours3, Scalar(255));
	GaussianBlur(matMask, matMask, Size(9, 9), 0);

	// apply mask
	Mat matFinal = matResizedOrg;
	for (int i = 0; i < matResizedOrg.rows; i++) {
		for (int j = 0; j < matResizedOrg.cols; j++) {
			double dOpacity = (double)matMask.at<uchar>(i, j) / 255;
			matFinal.at<Vec3b>(i, j)[0] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[0] * dOpacity);
			matFinal.at<Vec3b>(i, j)[1] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[1] * dOpacity);
			matFinal.at<Vec3b>(i, j)[2] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[2] * dOpacity);

		}
	}

	//imshow("VDiff", matVDiff);

	//imshow("ProcPolor", matProcVPolor);

	if(!bTest){
		imshow("Result", matFinal);
		waitKey();
	}

	return DT_SUCCESS;
}


/**
* @brief Process image to detect edge of tongue - 2nd algorithm

* Process image to detect edge of tongue ### algorithm 2

* @param filePath the path of image file to be processed
* @param bTest 0: normal mode(default), 1: unit test mode(should not allow any output)
* @return DT_SUCCESS: finished successfully, DT_FILENOTEXIST: file not found, DT_FAIL: failure
* @author Pai Jin
* @date 2017/7/20 (demo version for testing algorithm)
*/
int tongueDetectionAlgorithmUpgrade(const char* filePath, int bTest) {
	int i = 0;
	int j = 0;
	Mat matOrg = imread(filePath);
	if(matOrg.rows == 0){
		if(!bTest) printf("Can not open the file: %s\n", filePath);
		return DT_FILENOTEXIST;
	}
	Mat matResizedOrg;
	resize(matOrg, matResizedOrg, Size(matOrg.cols / 2, matOrg.rows / 2));
	GaussianBlur(matResizedOrg, matResizedOrg, Size(5, 5), 1);

	Mat matOrgHSV;
	cvtColor(matResizedOrg, matOrgHSV, CV_BGR2HSV);

	split(matOrgHSV, g_tempChannels);

	if(!bTest) imshow("V", matResizedOrg);

	Mat matVPolor;
	Point2f center = Point2f(matResizedOrg.cols / 2, matResizedOrg.rows / 2);
	double radius = (matResizedOrg.cols > matResizedOrg.rows) ? matResizedOrg.cols / 2 : matResizedOrg.rows / 2;
	linearPolar(g_tempChannels[2], matVPolor, center, radius, INTER_LINEAR + WARP_FILL_OUTLIERS);

	resize(matVPolor, matVPolor, Size((int)radius / 2, matVPolor.rows));

	Mat matVPolorDiff;
	differentiate(matVPolor, matVPolorDiff, 2, 4);

	Mat matElem = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1, 1)); // erode amount: 2
	erode(matVPolorDiff, matVPolorDiff, matElem);

	Mat matVPolorDiffResv = matVPolorDiff.clone();

	cvtColor(matVPolor, matVPolor, CV_GRAY2BGR);

	Mat matSimplifiedPolor = Mat::zeros(Size(matVPolor.cols, matVPolor.rows), CV_8UC1);

	findContours(matVPolorDiffResv, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	g_contours2.clear();
	g_boundingRects.clear();
	for (i = 0; i < g_contours1.size(); i++) {
		Rect rt = boundingRect(g_contours1[i]);
		if (rt.height > matSimplifiedPolor.rows / 16) {

			approxPolyDP(g_contours1[i], g_tempContour1, 4, true);

			//polylines(matVPolor, g_contours1[i], true, Scalar(0, 0, 255));
			polylines(matVPolor, g_tempContour1, true, Scalar(0, 0, 255));

			//g_contours2.push_back(g_contours1[i]);
			g_contours2.push_back(g_tempContour1);
			g_boundingRects.push_back(rt);
		}
	}

	/*for (int i = g_contours2.size() - 1; i >= 0; i--) {
		int del = 0;
		for (int k = 0; k < g_contours2.size(); k++) {
			if (i == k) continue;
			if (g_boundingRects[i].y > g_boundingRects[k].y && g_boundingRects[i].br().y < g_boundingRects[k].br().y) {
				del = 1;
				break;
			}
		}
		if (del) {
			g_contours2.erase(g_contours2.begin() + i);
			g_boundingRects.erase(g_boundingRects.begin() + i);
		}
	}*/


	int maxId = -1;
	int maxH = 0;
	for (i = 0; i < g_contours2.size(); i++) {
		Rect rti = g_boundingRects[i];
		if (rti.height > maxH) {
			maxH = rti.height;
			maxId = i;
		}
	}

	//if (maxH < matSimplifiedPolor.rows * 8 / 10) {
		for (int i = 0; i < g_contours2.size(); i++) {
			fillPoly(matSimplifiedPolor, g_contours2, Scalar(255));
		}
	//}
	/*else {
		g_contours3.clear();
		g_contours3.push_back(g_contours2[maxId]);
		fillPoly(matSimplifiedPolor, g_contours3, Scalar(255));
	}*/

	dilate(matSimplifiedPolor, matSimplifiedPolor, matElem);

	int* marks = new int[matSimplifiedPolor.rows];

	vector<Point> ptMarks;
	int iScanStepY = matSimplifiedPolor.rows / 60;
	int iScanStartX = matSimplifiedPolor.cols / 2;
	int iScanEndX = matSimplifiedPolor.cols * 7 / 8;
	int bFirst = 1;
	int lastX = matSimplifiedPolor.cols * 3 / 4;
	for (i = 0; i < 60; i++) {
		int rowTemp = i * iScanStepY;
		int lpos = -1;
		//vector<int> openList;
		for (j = iScanStartX; j < iScanEndX; j++) {
			//if (matSimplifiedPolor.at<uchar>(rowTemp, j) == 255 && matSimplifiedPolor.at<uchar>(rowTemp, j - 1) == 0) {
			if (matSimplifiedPolor.at<uchar>(rowTemp, j) == 255) {
				lpos = j;
				break;
			}
		}

		int minDiff = 1000;
		if (lpos > 0) {
			
			if (bFirst) {
				bFirst = 0;
				iScanStartX = matSimplifiedPolor.cols / 4;
				iScanEndX = matSimplifiedPolor.cols * 7 / 8;
			}
			lastX = lpos;
			ptMarks.push_back(Point(lpos, rowTemp));
		}
		
	}

	cvtColor(matSimplifiedPolor, matSimplifiedPolor, CV_GRAY2RGB);
	
	bFirst = 1;
	vector<Point> ptMarksSmooth;
	if (ptMarks.size() > 3) {
		ptMarksSmooth.push_back(ptMarks.front());
		for (i = 1; i < ptMarks.size() - 1; i++) {
			int x = (ptMarks[i - 1].x + ptMarks[i].x + ptMarks[i + 1].x) / 3;
			ptMarksSmooth.push_back(Point(x, ptMarks[i].y));
			line(matSimplifiedPolor, ptMarksSmooth[i-1], ptMarksSmooth[i], Scalar(0, 0, 255));
		}
		ptMarksSmooth.push_back(ptMarks.back());
	}
	else {
		return  DT_FAIL;
	}

	Mat matPolorEdgeSmooth = Mat::zeros(matSimplifiedPolor.rows, matSimplifiedPolor.cols, CV_8UC1);
	Point ptLast = Point(ptMarksSmooth.back().x, ptMarksSmooth.back().y - matSimplifiedPolor.rows);
	line(matPolorEdgeSmooth, ptLast, ptMarksSmooth[0], Scalar(255), 3);
	for (i = 0; i < ptMarksSmooth.size() - 1; i++) {
		line(matPolorEdgeSmooth, ptMarksSmooth[i], ptMarksSmooth[i + 1], Scalar(255), 3);
	}
	ptLast = Point(ptMarksSmooth.front().x, matSimplifiedPolor.rows + ptMarksSmooth.front().y);
	line(matPolorEdgeSmooth, ptMarksSmooth.back(), ptLast, Scalar(255), 3);

	//imshow("PolorEdge", matPolorEdgeSmooth);

	int density = 1;

	for (i = 0; i < matPolorEdgeSmooth.rows; i++) {
		marks[i] = -1;
		for (j = 0; j < matPolorEdgeSmooth.cols; j++) {
			if (matPolorEdgeSmooth.at<uchar>(i, j) == 255) {
				marks[i] = j + 5;
				break;
			}
		}
	}
	

	g_contours3.clear();
	g_tempContour2.clear();
	double m = 2 * 3.141592 / matSimplifiedPolor.rows;
	for (int i = 0; i < matSimplifiedPolor.rows; i++) {
		if (marks[i] >= 0) {
			double angle = m*i;
			double r = radius * marks[i] / matSimplifiedPolor.cols;
			int x = matResizedOrg.cols / 2 + (int)(cos(angle)*r);
			int y = matResizedOrg.rows / 2 + (int)(sin(angle)*r);
			if (i == 108) {
				int k = 0;
			}
			g_tempContour2.push_back(Point(x, y));
		}
	}


	delete[] marks;

	g_tempContour3.clear();
	int smooth = 5;
	for (int i = 0; i < g_tempContour2.size(); i++) {

		int x = 0;
		int y = 0;
		for (int j = -smooth; j <= smooth; j++) {
			int id = (i + j + g_tempContour2.size()) % g_tempContour2.size();
			x += g_tempContour2[id].x;
			y += g_tempContour2[id].y;
		}

		x /= (smooth * 2 + 1);
		y /= (smooth * 2 + 1);

		g_tempContour3.push_back(Point(x, y));
	}

	g_contours3.push_back(g_tempContour3);

	Mat mask = Mat::zeros(matResizedOrg.rows, matResizedOrg.cols, CV_8UC1);
	fillPoly(mask, g_contours3, Scalar(255));

	GaussianBlur(mask, mask, Size(9, 9), 0);

	Mat res = matResizedOrg;
	for (int i = 0; i < matResizedOrg.rows; i++) {
		for (int j = 0; j < matResizedOrg.cols; j++) {
			double rate = (double)mask.at<uchar>(i, j) / 255;
			res.at<Vec3b>(i, j)[0] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[0] * rate);
			res.at<Vec3b>(i, j)[1] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[1] * rate);
			res.at<Vec3b>(i, j)[2] = (uchar)((int)matResizedOrg.at<Vec3b>(i, j)[2] * rate);

		}
	}

	//imshow("VPolor", matVPolorDiff);
	//imshow("Polor", matSimplifiedPolor);

	if(!bTest) {
		imwrite("res.jpg", res);
		imshow("res", res);
		waitKey();
	}

	return DT_SUCCESS;
}






