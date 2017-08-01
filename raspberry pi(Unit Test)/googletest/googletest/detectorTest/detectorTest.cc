#include <limits.h>
#include "detector.h"
#include <gtest/gtest.h> 

/**
* detectorTest case
* usage: implement unit test
* @author RG Jong
* @date 2017/8/1
*/

TEST(DetectTongue, EXISTINGFILE) {
  EXPECT_EQ(DT_SUCCESS, tongueDetectionAlgorithm("images/image1.jpg", 1));
}
TEST(DetectTongue, NOTEXISTINGFILE) {
  EXPECT_EQ(DT_FILENOTEXIST, tongueDetectionAlgorithm("non-image.jpg", 1));
}
TEST(DetectTongueUpgrade, EXISTINGFILE) {
  EXPECT_EQ(DT_SUCCESS, tongueDetectionAlgorithmUpgrade("images/image1.jpg", 1));
}
TEST(DetectTongueUpgrade, NOTEXISTINGFILE) {
  EXPECT_EQ(DT_FILENOTEXIST, tongueDetectionAlgorithmUpgrade("non-image.jpg", 1));
}

int main(int argc, char **argv) {
//printf("Running main() from gtest_main.cc\n");
testing::InitGoogleTest(&argc,argv);
return RUN_ALL_TESTS();
}

