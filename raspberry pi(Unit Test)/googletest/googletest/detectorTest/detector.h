#ifndef GTEST_DETECTOR_H_
#define GTEST_DETECTOR_H_
#endif

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>

using namespace cv;

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
Mat erosion(Mat mat, int size);

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
Mat dilation(Mat mat, int size);

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
Mat removeNoiseContaur(Mat matNoise, int thresold);


/**
* @brief Get skin mask from saturation image

* @param Mat: saturation image
* @return Mat: mask image of skin area
* @author Pai Jin
* @date 2017/9/14
*/
Mat getSkinMask(Mat matSat);

/**
* @brief Tongue detection function

* Read image file from filepath, detect tongue area and output to image file.

* @param const char*: filepath of image file
* @return bool: true: success, false: failed
* @author Pai Jin
* @date 2017/9/16
*/
bool tongueDetection(const char* filePath);