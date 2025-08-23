#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <tuple>
#include <filesystem>
#include <chrono>   // time measurement
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using Detection = std::tuple<cv::Point, double>;

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
 * @brief Corrects the illumination of a gray scale image using a Gaussian filter
 * 
 * @param I gray scale image
 * 
 * @return The resulting image after illumination correction
 */
cv::Mat correct_illumination(cv::Mat I);

/**
 * @brief Apply the transformation T(r) = {if r <= T then L/T *r else L}
 * 
 * @param I gray scale image
 * @param threshold threshold T
 * 
 * @return The resulting image after the transformation
**/
cv::Mat contrast_stretching(cv::Mat I, int threshold, int max_coin_value = 200);

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

/**
 * @brief Applies preprocessing to the images in the test set, aims to avoid too bright areas
 * 
 * @param images Vector of input images
 * @param T threshold for contrast stretching
 * @param s size of the Gaussian kernel
 * @param sigma standard deviation for GaussianBlur
 * 
 * @return Vector of preprocessed images
 */
 std::vector<cv::Mat> preprocess_images_test(const std::vector<cv::Mat>& images, float T, int s, float sigma);


// ----- TEMPLATE MATCHING -----

/**
 * @brief Finds all the matches which have values above a certain threshold
 *
 * @param result result matrix (output of matchTemplate)
 * @param threshold threshold value
 * 
 * @return Vector of (position, value) tuples
 */
std::vector<std::tuple<cv::Point, float>> get_positions_and_values_above_threshold(const cv::Mat& result, double threshold);
/**
 * @brief Adds the new_point to the list if there is no nearby point within min_distance. 
 * If the new_point has a higher confidence value, it replaces the existing point, otherwise it is discarded
 * 
 * @param new_point the point to add
 * @param points vector of points to check
 * @param min_distance minimum distance to consider a point as "near"
 * 
 * @return True if a nearby point exists, false otherwise
 */
bool add_near_point(const std::tuple<cv::Point, float, float, std::string>& new_point, std::vector<std::tuple<cv::Point, float, float, std::string>>& points, double min_distance);

/**
 * @brief Creates rotated versions of a template image
 *
 * @param templ template image to rotate
 * @param num_rotations number of rotations
 * 
 * @return Vector of rotated images
 */
std::vector<cv::Mat> rotate_template(const cv::Mat& templ, const int num_rotations);

#endif // MAIN_H
