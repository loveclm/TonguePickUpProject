
// CodeRecogDlg.cpp : implementation file
//

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"

#include <vector>
#include <string>
#include <dirent.h>

using namespace std;
using namespace cv;


/*****************************************************************************
Main part of algorithm
******************************************************************************/

// Global variables using to contour operations.
vector<Mat> g_tempChannels;
vector<vector<Point>> g_contours1;
vector<vector<Point>> g_contours2;
vector<vector<Point>> g_contours3;
vector<Point> g_tempContour1;
vector<Point> g_tempContour2;
vector<Rect> g_boundingRects;

/**
* @brief Erosion operation

* Erosion = erode + dilate
* This operation used to remove noise.

* @param Mat: sourse image matrix
* @param int: erode and dilate operation size.
* @return Mat: Dest image matrix
* @author Pai Jin
* @date 2017/9/12
*/
Mat erosion(Mat mat, int size)
{
	Mat erosion_dst;

	if (mat.data == NULL) {
		std::cout << "erosion(): mat parameter is invalid." << std::endl;
		return mat;
	}
	if (size < 0) {
		std::cout << "erosion(): size parameter must be positive value." << std::endl;
		return erosion_dst;
	}

	int erosion_type = MORPH_RECT;
	int erosion_size = size;

	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	erode(mat, erosion_dst, element);
	return erosion_dst;
}


/**
* @brief Dilation operation

* Dilation = dilate + erode
* This operation used to remove holds.

* @param Mat: sourse image matrix
* @param int: erode and dilate operation size.
* @return Mat: Dest image matrix
* @author Pai Jin
* @date 2017/9/12
*/
Mat dilation(Mat mat, int size)
{
	Mat dilation_dst;

	if (mat.data == NULL) {
		std::cout << "dilation(): mat parameter is invalid." << std::endl;
		return mat;
	}
	if (size < 0) {
		std::cout << "dilation(): size parameter must be positive value." << std::endl;
		return dilation_dst;
	}

	int dilation_type = MORPH_RECT;
	int dilation_size = size;

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	/// Apply the dilation operation
	dilate(mat, dilation_dst, element);
	return dilation_dst;
}

/**
* @brief Remove seperated noise contours from mask images

* Mask image from thresolding include seperated noise area.
* To remove these noise areas, at first find all contourse and using bounding rect
* to remove noise area.

* @param Mat: sourse image matrix that included noise area
* @param int: thresold value.
* @return Mat: Dest image matrix that removed noise area
* @author Pai Jin
* @date 2017/9/16
*/
Mat removeNoiseContaur(Mat matNoise, int thresold) {
	Mat matRet;

	if (matNoise.data == NULL) {
		std::cout << "removeNoiseContaur(): matNoise parameter is invalid." << std::endl;
		return matRet;
	}
	if (thresold < 1) {
		std::cout << "removeNoiseContaur(): thresold parameter must be great than zero." << std::endl;
		return matRet;
	}

	// find all contours
	findContours(matNoise, g_contours1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	// if contour's bounding rect is small than thresold, this contour regard as noise contour and will be removed.
	g_contours2.clear();
	Rect rtB = boundingRect(g_contours1[g_contours1.size() - 1]);
	for (int i = 0; i < g_contours1.size(); i++) {



		// Get contour's bounding rect
		Rect rt = boundingRect(g_contours1[i]);

		// get points of contour
		approxPolyDP(g_contours1[i], g_tempContour1, 1, true);

		// The contour that is small than thresold will be removed.
		// and other contour's points will be smoothed.
		if (rt.height > matNoise.rows / thresold && g_tempContour1.size() > 100) {

			// smooth processing
			int smooth = 5;
			for (int i = 0; i < g_tempContour1.size(); i++) {

				int x = 0;
				int y = 0;
				int count = 0;

				if (g_tempContour1[i].y > 10) {
					for (int j = -smooth; j <= smooth; j++) {
						int id = (i + j + g_tempContour1.size()) % g_tempContour1.size();
						if ((abs(g_tempContour1[i].x - g_tempContour1[id].x) > 50 ||
							g_tempContour1[id].y < 10) &&
							id != i) {
							continue;
						}
						x += g_tempContour1[id].x;
						y += g_tempContour1[id].y;
						count++;
					}

					x /= count;
					y /= count;
				}
				else {
					x = g_tempContour1[i].x;
					y = g_tempContour1[i].y;
				}

				g_tempContour2.push_back(Point(x, y));
			}

			g_contours2.push_back(g_tempContour1);
		}
	}

	// Generate mask from contours that removed noise area.
	matRet = Mat::zeros(Size(matNoise.cols, matNoise.rows), CV_8UC1);
	fillPoly(matRet, g_contours2, Scalar(255));

	return matRet;
}


/**
* @brief Get skin mask from saturation image

* @param Mat: saturation image
* @return Mat: mask image of skin area
* @author Pai Jin
* @date 2017/9/14
*/
Mat getSkinMask(Mat matSat) {
	Mat matSkin, matSkinMask;

	if (matSat.data == NULL) {
		std::cout << "getSkinMask(): matsat parameter is invalid." << std::endl;
		return matSkinMask;
	}

	// To remove grey pixels, thresolding saturation image by OTSU method.
	threshold(matSat, matSkin, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// because grey pixels area generally consists of two parts(left and right area), create left and right mask.
	cv::Mat maskL = cv::Mat::zeros(matSat.rows + 2, matSat.cols + 2, CV_8U);
	cv::Mat maskR = cv::Mat::zeros(matSat.rows + 2, matSat.cols + 2, CV_8U);
	cv::floodFill(matSkin, maskL, cv::Point(0, 0), 255, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
	cv::floodFill(matSkin, maskR, cv::Point(matSat.cols - 2, matSat.rows - 2), 255, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);

	//Left and Right area merge to one area. and rescailing
	maskL = maskL + maskR;
	maskL = 255 - maskL;
	maskL(Range(1, maskL.rows - 1), Range(1, maskL.cols - 1)).copyTo(matSkinMask);

	// Remove noise.
	matSkinMask = removeNoiseContaur(matSkinMask, 4);

	if (matSkinMask.data == NULL) {
		std::cout << "removeNoiseContaur() function failed." << std::endl;
		return matSkinMask;
	}

	return matSkinMask;
}


/**
* @brief get difference matrix

* @param Mat: original image
* @param Mat: destination image
* @param int: distance of two points
* @param float: middle point
* @author Pai Jin
* @date 2017/9/14
*/void differentiate(Mat& src, Mat& dst, int step, int thres) {
	int w, h;
	w = src.cols;
	h = src.rows;
	dst = Mat::zeros(Size(w, h), src.type());
	for (int i = 0; i < h; i++) {
		for (int j = step; j < w - step; j++) {
			if (src.at<unsigned char>(i, j + step) > 0 && src.at<unsigned char>(i, j - step) > 0) {
				int diff = abs(src.at<unsigned char>(i, j - step) - src.at<unsigned char>(i, j + step));
				dst.at<unsigned char>(i, j) = (diff > thres) ? 255 : 0;
			}
		}
	}
}


/**
* @brief Image emphasis

* @param Mat: original image
* @param int: min value
* @param int: max value
* @param float: middle point
* @return Mat: emphasis image
* @author Pai Jin
* @date 2017/9/14
*/
Mat levelProcessing(Mat matOrg, int minVal, int maxVal, float mid) {
	for (int i = 0; i < matOrg.rows; i++) {
		for (int j = 0; j < matOrg.cols; j++) {
			if (matOrg.at<unsigned char>(i, j) < minVal) {
				matOrg.at<unsigned char>(i, j) = 0;
			}
			else if (matOrg.at<unsigned char>(i, j) > maxVal) {
				matOrg.at<unsigned char>(i, j) = 255;
			}
			else {
				float vX = matOrg.at<unsigned char>(i, j);
				float vY = 0.0f;

				int midX = float(maxVal - minVal) * mid + float(minVal);
				int midY = 255.0f * float(1.0f - mid);


				if (vX < midX) {
					vY = float(vX - minVal) / float(midX - minVal) * midY;
				}
				else {
					vY = float(vX - midX) / float(maxVal - midX) * float(255 - midY) + midY;
				}
				matOrg.at<unsigned char>(i, j) = int(vY);
			}
		}
	}

	return matOrg;
}


/**
* @brief Get tongue mask from saturation image

* @param Mat: original image
* @return Mat: mask image of tongue area
* @author Pai Jin
* @date 2017/9/14
*/
Mat getTangueMask(Mat matOrg) {
	// read image from file to opencv mat
	//Mat matOrg = imread(filePath);
	Mat matResizedOrg = matOrg;

	// convert image from RGB mode to HSV mode
	Mat matOrgHSV;
	cvtColor(matResizedOrg, matOrgHSV, CV_BGR2HSV);

	// split image into channels to get Value channel
	split(matOrgHSV, g_tempChannels);

	Mat matLevel;
	matLevel = levelProcessing(g_tempChannels[2], 100, 165, 0.5f);
	//imshow("matLevel", matLevel);

	// transform image from vertical coordinate system into polar coordinate system
	Mat matVPolar;
	Point2f center = Point2f(matResizedOrg.cols / 2, matResizedOrg.rows / 2);
	double radius = (matResizedOrg.cols > matResizedOrg.rows) ? matResizedOrg.cols / 2 : matResizedOrg.rows / 2;
	linearPolar(matLevel, matVPolar, center, radius, INTER_LINEAR + WARP_FILL_OUTLIERS);

	resize(matVPolar, matVPolar, Size((int)radius / 2, matVPolar.rows));

	// differentiate transformed image horizontally
	Mat matVPolarDiff;
	differentiate(matVPolar, matVPolarDiff, 2, 25);
	//imshow("matVPolarDiff", matVPolarDiff);

	// remove noises
	Mat matElem = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(1, 1)); // erode amount: 2
	matVPolarDiff = dilation(matVPolarDiff, 1);
	matVPolarDiff = erosion(matVPolarDiff, 1);
	//imshow("matVPolarDiff1", matVPolarDiff);

	Mat matVPolarDiffResv = matVPolarDiff.clone();

	cvtColor(matVPolar, matVPolar, CV_GRAY2BGR);

	// simplify image in polar coordinate system
	Mat matSimplifiedPolar = Mat::zeros(Size(matVPolar.cols, matVPolar.rows), CV_8UC1);

	// find all contours
	findContours(matVPolarDiffResv, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// choose contours the height of which is greater than one 16th of the height of the image
	g_contours2.clear();
	g_boundingRects.clear();
	for (int i = 0; i < g_contours1.size(); i++) {
		Rect rt = boundingRect(g_contours1[i]);
		if (rt.height > matSimplifiedPolar.rows / 16) {

			approxPolyDP(g_contours1[i], g_tempContour1, 1, true);

			g_contours2.push_back(g_tempContour1);
			g_boundingRects.push_back(rt);
		}
	}

	// generate new black-white image of polar coordinate system
	for (int i = 0; i < g_contours2.size(); i++) {
		fillPoly(matSimplifiedPolar, g_contours2, Scalar(255));
	}


	//imshow("matSimplifiedPolar0", matSimplifiedPolar);


	// guess the edge of tongue from the differentiated and simplified image in polar coordinate system
	int* marks = new int[matSimplifiedPolar.rows];

	vector<Point> ptMarks;
	float iScanStepY = (matSimplifiedPolar.rows - 5) / 80.0f;
	int iScanStartX = matSimplifiedPolar.cols / 12;
	int iScanEndX = matSimplifiedPolar.cols * 11 / 12;
	int bFirst = 1;
	int lastX = matSimplifiedPolar.cols * 11 / 12;
	int exceptionCount = 0;
	for (int i = 0; i < 80; i++) {
		int rowTemp = 5 + i * iScanStepY;
		int lpos = -1;
		for (int j = iScanStartX; j < iScanEndX; j++) {
			if (matSimplifiedPolar.at<unsigned char>(rowTemp, j) == 255) {
				lpos = j;
				break;
			}
		}

		int minDiff = 1000;
		if (lpos > 0) {

			if (bFirst) {
				bFirst = 0;
				iScanStartX = matSimplifiedPolar.cols / 12;
				iScanEndX = matSimplifiedPolar.cols * 11 / 12;
			}

			if (lastX < matSimplifiedPolar.cols * 11 / 12 && abs(lastX - lpos) > matSimplifiedPolar.cols / 5) {
				exceptionCount++;
				if (exceptionCount > 5) {
					Mat matRet;
					return matRet;
				}
				continue;
			}


			lastX = lpos;
			ptMarks.push_back(Point(lpos + 5, rowTemp));
		}

	}

	cvtColor(matSimplifiedPolar, matSimplifiedPolar, CV_GRAY2RGB);

	//imshow("matSimplifiedPolar", matSimplifiedPolar);

	bFirst = 1;
	vector<Point> ptMarksSmooth;
	if (ptMarks.size() > 3) {
		ptMarksSmooth.push_back(ptMarks.front());
		for (int i = 1; i < ptMarks.size() - 1; i++) {
			int x = (ptMarks[i - 1].x + ptMarks[i].x + ptMarks[i + 1].x) / 3;
			ptMarksSmooth.push_back(Point(x, ptMarks[i].y));
			line(matSimplifiedPolar, ptMarksSmooth[i - 1], ptMarksSmooth[i], Scalar(0, 0, 255));
		}
		ptMarksSmooth.push_back(ptMarks.back());
	}
	else {
		Mat matRet;
		return matRet;
	}

	Mat matPolarEdgeSmooth = Mat::zeros(matSimplifiedPolar.rows, matSimplifiedPolar.cols, CV_8UC1);
	Point ptLast = Point(ptMarksSmooth.back().x, ptMarksSmooth.back().y - matSimplifiedPolar.rows);
	line(matPolarEdgeSmooth, ptLast, ptMarksSmooth[0], Scalar(255), 3);
	for (int i = 0; i < ptMarksSmooth.size() - 1; i++) {
		line(matPolarEdgeSmooth, ptMarksSmooth[i], ptMarksSmooth[i + 1], Scalar(255), 3);
	}
	ptLast = Point(ptMarksSmooth.front().x, matSimplifiedPolar.rows + ptMarksSmooth.front().y);
	line(matPolarEdgeSmooth, ptMarksSmooth.back(), ptLast, Scalar(255), 3);

	//imshow("matPolarEdgeSmooth", matPolarEdgeSmooth);

	int density = 1;

	for (int i = 0; i < matPolarEdgeSmooth.rows; i++) {
		marks[i] = -1;
		for (int j = 0; j < matPolarEdgeSmooth.cols; j++) {
			if (matPolarEdgeSmooth.at<unsigned char>(i, j) == 255) {
				marks[i] = j + 5;
				break;
			}
		}
	}

	// generate polyline in original coordinate system from the detected edge in polar coordinate system
	g_contours3.clear();
	g_tempContour2.clear();
	double m = 2 * 3.141592 / matSimplifiedPolar.rows;
	for (int i = 0; i < matSimplifiedPolar.rows; i++) {
		if (marks[i] >= 0) {
			double angle = m*i;
			double r = radius * marks[i] / matSimplifiedPolar.cols;
			int x = matResizedOrg.cols / 2 + (int)(cos(angle)*r);
			int y = matResizedOrg.rows / 2 + (int)(sin(angle)*r);
			if (i == 108) {
				int k = 0;
			}
			g_tempContour2.push_back(Point(x, y));
		}
	}

	delete[] marks;

	g_contours3.push_back(g_tempContour2);

	// create mask image with black and white
	Mat mask = Mat::zeros(matResizedOrg.rows, matResizedOrg.cols, CV_8UC1);
	fillPoly(mask, g_contours3, Scalar(255));

	//imshow("Tangue Mask", mask);

	return mask;
}



/**
* @brief Display text information

* @param Mat: img
* @param string: txt
* @author Pai Jin
* @date 2017/9/16
*/
void displayText(Mat img, string txt) {
	cv::Point myPoint;
	myPoint.x = 100;
	myPoint.y = 100;

	/// Font Face
	int myFontFace = 2;

	/// Font Scale
	double myFontScale = 1.6;

	cv::putText(img, txt, myPoint, myFontFace, myFontScale, Scalar(0, 0, 255));
}



/**
* @brief Skin detection function

* Read image file from filepath, detect tongue area and output to image file.

* @param const char*: filepath of image file
* @return bool: true: success, false: failed
* @author Pai Jin
* @date 2017/9/16
*/
bool skinDetection(const char* filePath) {
	if (filePath == NULL) {
		std::cout << "Please input filename." << filePath << std::endl;
		return false;
	}
	// read image from file to opencv mat
	Mat matOrg = imread(filePath);

	if (matOrg.data == NULL) {
		std::cout << "counld not read " << filePath << "." << std::endl;
		return false;
	}

	// initialize global variables
	g_tempChannels.clear();
	g_contours1.clear();
	g_contours2.clear();
	g_contours3.clear();
	g_tempContour1.clear();
	g_tempContour2.clear();

	/*
	// resize image
	Mat matResizedOrg;
	resize(matOrg, matResizedOrg, Size(matOrg.cols / 2, matOrg.rows / 2));
	*/

	// convert original image to HSV image
	Mat matOrgHSV;
	cvtColor(matOrg, matOrgHSV, CV_BGR2HSV);

	// split image into channels to get Value channel
	split(matOrgHSV, g_tempChannels);

	// get skin mask
	Mat matSkinMask = getSkinMask(g_tempChannels[1]);
	if (matSkinMask.data == NULL) {
		std::cout << "tongueDetection() function failed." << std::endl;
		return false;
	}

	// get skin image
	Mat matSkin;
	matOrg.copyTo(matSkin, matSkinMask);
	//imshow("Skin", matSkin);

	std::string fpath1 = filePath;
	fpath1.replace(fpath1.length() - 4, fpath1.length() - 1, "_skin.jpg");
	imwrite(fpath1.c_str(), matSkin);

	return true;
}


/**
* @brief Tongue detection function

* Read image file from filepath, detect tongue area and output to image file.

* @param const char*: filepath of image file
* @return bool: true: success, false: failed
* @author Pai Jin
* @date 2017/9/16
*/
bool tongueDetection(const char* filePath) {
	if (filePath == NULL) {
		std::cout << "Please input filename." << filePath << std::endl;
		return false;
	}
	// read image from file to opencv mat
	Mat matOrg = imread(filePath);

	if (matOrg.data == NULL) {
		std::cout << "counld not read " << filePath << "." << std::endl;
		return false;
	}

	// initialize global variables
	g_tempChannels.clear();
	g_contours1.clear();
	g_contours2.clear();
	g_contours3.clear();
	g_tempContour1.clear();
	g_tempContour2.clear();

	/*
	/// resize image
	Mat matResizedOrg;
	resize(matOrg, matOrg, Size(matOrg.cols / 2, matOrg.rows / 2));
	*/

	Mat matTongueMask;
	matTongueMask = getTangueMask(matOrg);

	Mat matTongue;
	if (matTongueMask.data == NULL) {
		matTongue = Mat::zeros(matOrg.rows, matOrg.cols, CV_8UC3);
		displayText(matTongue, "Image not valid.");
	}
	else {
		matOrg.copyTo(matTongue, matTongueMask);
	}
	//imshow("matTongue", matTongue);

	std::string fpath1 = filePath;
	fpath1.replace(fpath1.length() - 4, fpath1.length() - 1, "_tongue.jpg");
	imwrite(fpath1.c_str(), matTongue);

	return true;
}



/**
* @brief Main function

* @date 2017/7/19
*/
int main(int argc, char* argv[]){
	if(argc < 3){
		printf("Usages :  %s %s %s\n", argv[0], argv[1], argv[2]);
		return 0;
	}

	if( argv[1][0] != '-' ){
		printf("Usages :  %s %s %s\n", argv[0], argv[1], argv[2]);
		return 0;
	}

	bool ret = false;
	int detectionMode = 0;

	switch (argv[1][1])
	{
	default:
		printf("Unknown option -%c\n\n", argv[1][0]);
		break;
	case 's':
		detectionMode = 1;
		break;
	case 't':
		detectionMode = 2;
		break;
	}

	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(argv[2]);
	if(dpdf != NULL){
		while( epdf = readdir( dpdf ) ){			
			std::string filename = argv[2];

			if( filename == "." || filename == ".."  ){
				continue;
			}

			filename += "/";
			filename += epdf->d_name;
			std::cout << filename << std::endl;			
			switch ( detectionMode ){
				case 1:
					ret = skinDetection(filename.c_str());
					break;
				case 2:
					ret = tongueDetection(filename.c_str());
					break;
			}
		}
	} 
	else {
		printf("failed getting files");
	}
	
	return 0;
}



