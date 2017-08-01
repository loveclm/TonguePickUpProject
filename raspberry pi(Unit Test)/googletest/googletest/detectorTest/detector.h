#ifndef GTEST_DETECTOR_H_
#define GTEST_DETECTOR_H_
#endif

#define DT_SUCCESS	0
#define DT_FAIL		1
#define DT_FILENOTEXIST 2



/**
* @brief Process image to detect edge of tongue

* Process image to detect edge of tongue

* @param filePath the path of image file to be processed
* @param bTest 0: normal mode(default), 1: unit test mode(should not allow any output)
* @return DT_SUCCESS: finished successfully, DT_FILENOTEXIST: file not found
* @author Pai Jin
* @date 2017/7/17 (demo version for testing algorithm)
*/
int tongueDetectionAlgorithm(const char* filePath, int bTest = 0);


/**
* @brief Process image to detect edge of tongue - 2nd algorithm

* Process image to detect edge of tongue ### algorithm 2

* @param filePath the path of image file to be processed
* @param bTest 0: normal mode(default), 1: unit test mode(should not allow any output)
* @return DT_SUCCESS: finished successfully, DT_FILENOTEXIST: file not found, DT_FAIL: failure
* @author Pai Jin
* @date 2017/7/20 (demo version for testing algorithm)
*/
int tongueDetectionAlgorithmUpgrade(const char* filePath, int bTest = 0);
