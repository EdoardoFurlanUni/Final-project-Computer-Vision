#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <tuple>

/**
 * @brief Display the histogram of a gray scale image
 * 
 * @param I gray scale image
 * @param bins number of bins in the histogram
 * @param name name of the window
 * 
 * @return A std::tuple containing: a double with the IoU and an integer with the accuracy.
**/
cv::Mat display_hist(cv::Mat I, int bins, std::string name);


#endif // MAIN_H
