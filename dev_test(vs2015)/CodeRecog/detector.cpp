
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
vector<Vec4i> hierarchy;

/**
* @brief Convert integer to string

* @param int: integer value
* @return string: converted sting
* @author Pai Jin
* @date 2017/9/16
*/
string iToS(int val) {
	stringstream ss;
	ss << val;
	string str = ss.str();
	return str;
}


/**
* @brief Display text information

* @param Mat: img
* @param string: txt
* @author Pai Jin
* @date 2017/9/16
*/
void displayText(Mat img, string txt, int x, int y) {
	cv::Point myPoint;
	myPoint.x = x;
	myPoint.y = y;

	/// Font Face
	int myFontFace = 2;

	/// Font Scale
	double myFontScale = 1;

	cv::putText(img, txt, myPoint, myFontFace, myFontScale, Scalar(0, 0, 255));
}


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
* @brief get difference matrix

* @param Mat: original image
* @param Mat: destination image
* @param int: distance of two points
* @param float: middle point
* @author Pai Jin
* @date 2017/9/14
*/
void differentiate(Mat& src, Mat& dst, int step, int thres) {
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
* @brief Remove noise area from mask image.

* @param Mat: original image
* @return Mat: mask image that removed noise area
* @author Pai Jin
* @date 2017/9/26
*/
Mat sliceTonguePolar(Mat matOrg) {
	Mat matRet = matOrg.clone();

	int prevStartPos = 0;
	for (int j = 2; j < matOrg.cols - 6; j++) {
		if (matOrg.at<unsigned char>(1, j) == 0) {
			prevStartPos = j;
			break;
		}
	}
	for (int i = 2; i < matOrg.rows; i++) {
		bool sliceStart = false;
		for (int j = 2; j < matOrg.cols - 6; j++) {
			if (sliceStart == false && matOrg.at<unsigned char>(i, j) == 0) {
				sliceStart = true;
				if ((j - prevStartPos) > matOrg.cols / 6) j = prevStartPos;

				prevStartPos = j;
			}
			if (sliceStart == true) {
				matRet.at<unsigned char>(i, j) = 0;
			}
		}
	}

	return matRet;
}


/**
* @brief when separating tongue area from polar translated image,
check validation of mask image.
* @param Mat: polar translated image
* @param Mat: mask image
* @return int: if mask include other not tongue area, return threshold value to separate other area.
* @author Pai Jin
* @date 2017/9/28
*/
int checkMaskValidation(Mat matLevel, Mat matMask) {
	Mat matTemp;
	matLevel.copyTo(matTemp, matMask);
	Mat matLevelCheck(matTemp, Range(matTemp.rows / 16, matTemp.rows * 7 / 16), Range(0, matTemp.cols));

	int th;
	for (th = 170; th < 255; th = th + 10) {
		Mat matT;
		threshold(matLevelCheck, matT, th, 255, CV_THRESH_BINARY);

		// find all contours
		findContours(matT, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		// choose contours the height of which is greater than one 16th of the height of the image
		g_contours2.clear();
		int count = 0;
		bool bHValid = false;
		bool bWValid = false;
		for (int i = 0; i < g_contours1.size(); i++) {
			Rect rt = boundingRect(g_contours1[i]);
			if (rt.width < 15 && rt.x < 15) {
				continue;
			}

			if (rt.height > matT.rows / 4) {
				count++;
			}
			if (rt.height > matT.rows * 7 / 8) {
				bHValid = true;
			}

			if (rt.height > matT.rows / 4) {
				approxPolyDP(g_contours1[i], g_tempContour1, 2, true);
				g_contours2.push_back(g_tempContour1);
			}
		}

		matTemp = Mat::zeros(Size(matT.cols, matT.rows), CV_8UC1);
		for (int i = 0; i < g_contours2.size(); i++) {
			fillPoly(matTemp, g_contours2, Scalar(255));
		}

		if (count > 1 && bHValid/* && bWValid*/) {
			imshow("checkFalse", matT);
			return th;
		}
	}

	return th;
}


/**
* @brief Blur mask image to smooth edge.

* @param Mat: original image
* @param Mat: mask image
* @param int: blur size
* @return Mat: blured mask image
* @author Pai Jin
* @date 2017/9/26
*/
Mat blurMasking(Mat matOrg, Mat matMask, int blurSize) {
	Mat matBlurMask, matBlur;
	matBlur = matOrg.clone();
	//matBlurMask = Mat::zeros(matMask.rows, matMask.cols, CV_8UC1);
	GaussianBlur(matMask, matBlurMask, Size(blurSize, blurSize), blurSize);

	for (int i = 0; i < matOrg.rows; i++) {
		for (int j = 0; j < matOrg.cols; j++) {
			matBlur.at<Vec3b>(i, j)[0] = matOrg.at<Vec3b>(i, j)[0] * matBlurMask.at<unsigned char>(i, j) / 255;
			matBlur.at<Vec3b>(i, j)[1] = matOrg.at<Vec3b>(i, j)[1] * matBlurMask.at<unsigned char>(i, j) / 255;
			matBlur.at<Vec3b>(i, j)[2] = matOrg.at<Vec3b>(i, j)[2] * matBlurMask.at<unsigned char>(i, j) / 255;
		}
	}

	return matBlur;
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
* @brief Get tongue mask from saturation image

* @param Mat: saturation image
* @return Mat: mask image of skin area
* @author Pai Jin
* @date 2017/9/14
*/
Mat getTangueMask(Mat matOrg) {
	Mat matLevel, matTemp, matVPolar, matVPolarOrg, matVPolarTemp;
	matTemp = matOrg.clone();
	Point2f center = Point2f(matOrg.cols / 2, matOrg.rows / 2);
	double radius = (matOrg.cols > matOrg.rows) ? matOrg.cols / 2 : matOrg.rows / 2;

	// loop for separating tongue area from image
	int levelThStep = 0;
	bool bOneSec = false;
	while (true) {
		// to emphasis tongue area, increase image's contrast.
		matTemp = matOrg.clone();
		matLevel = Mat::zeros(matTemp.rows, matTemp.cols, CV_8UC1);
		matLevel = levelProcessing(matTemp, 100 + levelThStep, 185, 0.5f);

		// translate image to polar coordination
		linearPolar(matLevel, matVPolarOrg, center, radius, INTER_LINEAR + WARP_FILL_OUTLIERS);

		// reduce image size to half.
		resize(matVPolarOrg, matVPolarOrg, Size((int)radius / 2, matVPolarOrg.rows));

		// to get tongue mask image, thresholding by OTSU
		threshold(matVPolarOrg, matVPolar, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		matVPolar = dilation(matVPolar, 1);
		matVPolar = erosion(matVPolar, 1);
		matVPolarTemp = matVPolar.clone();

		// when separating loop is failed, break loop
		if (bOneSec == true) break;

		// remove noise area from tongue mask image
		findContours(matVPolarTemp, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		g_contours2.clear();
		int count = 0;
		int wTh = 0;
		bool bForceDiv = false;
		for (int i = 0; i < g_contours1.size(); i++) {
			Rect rt = boundingRect(g_contours1[i]);
			if (rt.height > matVPolarTemp.rows / 6) {
				wTh = rt.width;
				if (wTh > matVPolarTemp.cols * 4 / 5) bForceDiv = true;
				count++;
			}
			if (rt.height > matVPolarTemp.rows / 2 && rt.x < matVPolarTemp.cols / 4) {
				approxPolyDP(g_contours1[i], g_tempContour1, 2, true);
				g_contours2.push_back(g_tempContour1);
			}
		}

		matVPolarTemp = Mat::zeros(Size(matVPolar.cols, matVPolar.rows), CV_8UC1);
		for (int i = 0; i < g_contours2.size(); i++) {
			fillPoly(matVPolarTemp, g_contours2, Scalar(255));
		}
		matVPolarTemp = dilation(matVPolarTemp, 1);

		// if tongue mask image is not vaild, increase contrast value.
		if (count == 1 || bForceDiv == true) {
			levelThStep += 15;
			if (levelThStep > 45) {
				levelThStep = 30;
				bOneSec = true;
			}
			continue;
		}
		else {
			break;
		}
	}

	// check validation of tongue mask image
	int validTh = checkMaskValidation(matVPolarOrg, matVPolarTemp);
	if (validTh > 210) {
		// if mask separating is not possible, get mask image by OTSU thresholding
		threshold(matVPolarOrg, matVPolar, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	}
	else {
		// if mask separating is possible, get mask image by validTh
		threshold(matVPolarOrg, matVPolar, validTh + 5, 255, CV_THRESH_BINARY);

		matVPolarTemp = matVPolar.clone();

		// remove noise area
		findContours(matVPolarTemp, g_contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		g_contours2.clear();
		for (int i = 0; i < g_contours1.size(); i++) {
			Rect rt = boundingRect(g_contours1[i]);
			if (rt.height > matVPolarTemp.rows / 2 && rt.x < matVPolarTemp.cols / 4) {
				approxPolyDP(g_contours1[i], g_tempContour1, 2, true);
				g_contours2.push_back(g_tempContour1);
			}
		}

		matVPolarTemp = Mat::zeros(Size(matVPolar.cols, matVPolar.rows), CV_8UC1);
		for (int i = 0; i < g_contours2.size(); i++) {
			fillPoly(matVPolarTemp, g_contours2, Scalar(255));
		}
		matVPolarTemp = dilation(matVPolarTemp, 3);
	}

	// remove big noise area
	Mat matSlicePolar;
	matSlicePolar = Mat::zeros(matVPolar.rows, matVPolar.cols, CV_8UC1);
	matSlicePolar = sliceTonguePolar(matVPolarTemp);

	// getting tongue mask's polyline
	vector<Point> ptMarks;
	int iScanStepY = (matSlicePolar.rows) / 40.0;
	for (int i = 0; i < 40; i++) {
		int rowTemp = 5 + i * iScanStepY;
		int lpos = -1;
		for (int j = matSlicePolar.cols - 20; j > 1; j--) {
			if (matSlicePolar.at<unsigned char>(rowTemp, j) == 255) {
				lpos = j - 5;
				break;
			}
		}

		if (lpos > 0) {
			ptMarks.push_back(Point(lpos, rowTemp));
		}
	}

	// smoothing tongue mask's polyline
	vector<Point> ptMarksSmooth;
	if (ptMarks.size() > 3) {
		ptMarksSmooth.push_back(ptMarks.front());
		for (int i = 1; i < ptMarks.size() - 1; i++) {
			int x = (ptMarks[i - 1].x + ptMarks[i].x + ptMarks[i + 1].x) / 3;
			ptMarksSmooth.push_back(Point(x, ptMarks[i].y));
			line(matSlicePolar, ptMarksSmooth[i - 1], ptMarksSmooth[i], Scalar(0, 0, 255));
		}
		ptMarksSmooth.push_back(ptMarks.back());
	}
	else {
		Mat matRet;
		return matRet;
	}

	Mat matPolarEdgeSmooth = Mat::zeros(matSlicePolar.rows, matSlicePolar.cols, CV_8UC1);
	for (int i = 0; i < ptMarksSmooth.size() - 1; i++) {
		line(matPolarEdgeSmooth, ptMarksSmooth[i], ptMarksSmooth[i + 1], Scalar(255), 3);
	}
	Point ptLast = Point(ptMarksSmooth[0].x, matSlicePolar.rows);
	line(matPolarEdgeSmooth, ptMarksSmooth.back(), ptLast, Scalar(255), 3);

	int* marks = new int[matSlicePolar.rows];
	for (int i = 0; i < matPolarEdgeSmooth.rows; i++) {
		marks[i] = -1;
		for (int j = 0; j < matPolarEdgeSmooth.cols; j++) {
			if (matPolarEdgeSmooth.at<unsigned char>(i, j) == 255) {
				marks[i] = j + 5;
				break;
			}
		}
	}

	// translate polyline in original coordinate system from smoothed polyline in polar coordinate system
	g_contours3.clear();
	g_tempContour2.clear();
	double m = 2 * 3.141592 / matSlicePolar.rows;
	for (int i = 0; i < matSlicePolar.rows; i++) {
		if (marks[i] >= 0) {
			double angle = m*i;
			double r = radius * marks[i] / matSlicePolar.cols;
			int x = matOrg.cols / 2 + (int)(cos(angle)*r);
			int y = matOrg.rows / 2 + (int)(sin(angle)*r);
			if (i == 108) {
				int k = 0;
			}
			g_tempContour2.push_back(Point(x, y));
		}
	}

	delete[] marks;

	// get tongue mask image in original coordinate.
	g_contours3.push_back(g_tempContour2);
	Mat mask = Mat::zeros(matOrg.rows, matOrg.cols, CV_8UC1);
	fillPoly(mask, g_contours3, Scalar(255));

	return mask;
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

	// save to file
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

	/// resize image to half
	Mat matResizedOrg;
	resize(matOrg, matResizedOrg, Size(matOrg.cols / 2, matOrg.rows / 2));

	// convert image from RGB mode to HSV mode	
	// split image into channels to get Value channel
	Mat matOrgHSV;
	cvtColor(matResizedOrg, matOrgHSV, CV_BGR2HSV);
	split(matOrgHSV, g_tempChannels);

	Mat matTongue, matTongueMask;
	matTongueMask = getTangueMask(g_tempChannels[2]);
	if (matTongueMask.data == NULL) {
		//matTongue = Mat::zeros(matOrg.rows, matOrg.cols, CV_8UC3);
		//matTongue = Scalar(255, 0, 0) + matTongue;
		//displayText(matTongue, "Image not valid.", 100, 100);
		return false;
	}
	else {
		// resize image to origne size
		resize(matTongueMask, matTongueMask, Size(matTongueMask.cols * 2, matTongueMask.rows * 2));

		// smoothin edge
		matTongue = blurMasking(matOrg, matTongueMask, 15);

		cvtColor(matTongue, matOrgHSV, CV_BGR2HSV);
		split(matOrgHSV, g_tempChannels);

		// getting histogram from tongue image
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true; bool accumulate = false;
		Mat h_hist, v_hist;
		calcHist(&g_tempChannels[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&g_tempChannels[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate);

		// checking validation of tongue image
		int v_histTh;
		v_histTh = matTongue.rows * matTongue.cols - matTongue.rows * matTongue.cols * 0.35 * 0.35;

		int hist = 0;
		for (int i = 0; i < v_hist.rows; i++) {
			hist += v_hist.at<float>(i);
		}

		int zeroHist = v_hist.at<float>(0);
		if (zeroHist > v_histTh) {
			//matTongue = Mat::zeros(matOrg.rows, matOrg.cols, CV_8UC3);
			//matTongue = Scalar(255, 0, 0) + matTongue;
			//displayText(matTongue, "Image not valid.", 100, 100);
			return false;
		}
	}

	// if tongue image is valid, save to file
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

			if( ret == false )
				std::cout << "Detection failed from " << filename << std::endl;
		}
	} 
	else {
		printf("failed getting files");
	}
	
	return 0;
}



