#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <tuple>

/**
 * @brief Apply the transformation T(r) = {if r <= T then L/T *r else L}
 * 
 * @param I gray scale image
 * @param threshold threshold T
 * 
 * @return The resulting image after the transformation
**/
cv::Mat contrast_streatching(cv::Mat I, int threshold);

/**
 * @brief Display the histogram of a gray scale image
 * 
 * @param I gray scale image
 * @param bins number of bins in the histogram
 * @param name name of the window
 * 
 * @return The displayed histogram
**/
cv::Mat display_hist(cv::Mat I, int bins, std::string name);


#endif // MAIN_H
