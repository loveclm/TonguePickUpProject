#include <limits.h>
#include "detector.h"
#include <gtest/gtest.h> 

/**
* detectorTest case
* usage: implement unit test
* @author RG Jong
* @date 2017/9/16
*/


TEST(ErosionTest, INVALIDPARAM) {
  Mat mat, matRes;	

  matRes = erosion(mat, 1);
  EXPECT_EQ(NULL, matRes.data);

  mat = Mat::zeros(Size(10, 10), CV_8UC1);
  matRes = erosion(mat, -1);
  EXPECT_EQ(NULL, matRes.data);
}

TEST(DilationTest, INVALIDPARAM) {
  Mat mat, matRes;	

  matRes = dilation(mat, 1);
  EXPECT_EQ(NULL, matRes.data);

  mat = Mat::zeros(Size(10, 10), CV_8UC1);
  matRes = dilation(mat, -1);
  EXPECT_EQ(NULL, matRes.data);
}

TEST(RemoveNoiseContaurTest, INVALIDPARAM) {
  Mat mat, matRes;	

  matRes = removeNoiseContaur(mat, 1);
  EXPECT_EQ(NULL, matRes.data);

  mat = Mat::zeros(Size(10, 10), CV_8UC1);
  matRes = removeNoiseContaur(mat, 0);
  EXPECT_EQ(NULL, matRes.data);

  matRes = removeNoiseContaur(mat, -5);
  EXPECT_EQ(NULL, matRes.data);
}

TEST(GetSkinMaskTest, INVALIDPARAM) {
  Mat mat, matRes;	

  matRes = getSkinMask(mat);
  EXPECT_EQ(NULL, matRes.data);
}

TEST(TongueDetectionTest, INVALIDPARAM) {
  bool res;	
  res = tongueDetection("non-image.jpg");
  EXPECT_EQ(false, res);
}

int main(int argc, char **argv) {
//printf("Running main() from gtest_main.cc\n");
testing::InitGoogleTest(&argc,argv);
return RUN_ALL_TESTS();
}

