
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
vector<Point> g_tempContour1;
vector<Point> g_tempContour2;

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

		// The contour that is small than thresold will be removed.
		// and other contour's points will be smoothed.
		if (rt.height > matNoise.rows / thresold) {

			// get points of contour
			approxPolyDP(g_contours1[i], g_tempContour1, 1, true);

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

			g_contours2.push_back(g_tempContour2);
		}
	}

	// Generate mask from contours that removed noise area.
	matRet = Mat::zeros(Size(matNoise.cols, matNoise.rows), CV_8UC1);
	fillPoly(matRet, g_contours2, Scalar(255));

	return matNoise;
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
	g_tempContour1.clear();
	g_tempContour2.clear();

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
* @brief Main function

* @date 2017/7/19
*/
int main(int argc, char* argv[]){
	if(argc < 2){
		printf("Usages :  %s filename\n", argv[0]);
	}

//std::string filePath = argv[1];
//std::cout << filePath << std::endl;

	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir(argv[1]);
	if(dpdf != NULL){
		while( epdf = readdir( dpdf ) ){			
			std::string filename = argv[1];

			if( filename == "." || filename == ".."  ){
				continue;
			}

			filename += "/";
			filename += epdf->d_name;
			std::cout << filename << std::endl;			
			bool ret = tongueDetection(filename.c_str());
		}
	} 
	else {
		printf("failed getting files");
	}


	return 0;
}



