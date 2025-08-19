#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <tuple>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// ----- FILE MANAGEMENT -----
/**
 * @brief Returns a vector of file paths in the given directory.
 * 
 * @param folder path to the directory
 * 
 * @return list of file paths
 */
std::vector<std::string> get_file_names(const std::string& folder);

/**
 * @brief Loads images from a folder
 * 
 * @param folders folder containing images
 * 
 * @return Vector of images
 */
std::vector<cv::Mat> load_images_from_folder(const std::string& folders);


// ----- PREPROCESSING -----
/**
 * @brief Apply the transformation T(r) = {if r <= T then L/T *r else L}
 * 
 * @param I gray scale image
 * @param threshold threshold T
 * 
 * @return The resulting image after the transformation
**/
cv::Mat contrast_stretching(cv::Mat I, int threshold);

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

/**
 * @brief Applies preprocessing to a vector of images
 * 
 * @param images Vector of input images
 * @param T threshold for contrast stretching
 * @param s size of the Gaussian kernel
 * @param sigma standard deviation for GaussianBlur
 * 
 * @return Vector of preprocessed images
 */
std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, float T, int s, float sigma);


#endif // MAIN_H
