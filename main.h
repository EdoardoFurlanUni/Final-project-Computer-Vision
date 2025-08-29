#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <tuple>
#include <filesystem>
#include <fstream>
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
 * @param folder folder containing images
 * @param flags flags for cv::imread (cv::IMREAD_GRAYSCALE, cv::IMREAD_COLOR_BGR, etc.)
 * 
 * @return Vector of images
 */
std::vector<cv::Mat> load_images_from_folder(const std::string& folder, int flags);

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
 * @brief Apply the transformation T(r) = {where x1 < r < x2 then r * (y2-y1)/(x2-x1)}
 * 
 * @param I gray scale image
 * @param points points of the intersections of the piece wise linear transformation (0,0) (100,100)... 
 * 
 * @return The resulting image after the transformation
 */
cv::Mat contrast_stretching(const cv::Mat& I, const std::vector<cv::Point2f>& points);

/**
 * @brief Display the histogram of a gray scale image
 * 
 * @param I gray scale image
 * @param bins number of bins in the histogram
 * @param name name of the window
 * 
 * @return The displayed histogram
*/
cv::Mat display_hist(cv::Mat I, int bins, std::string name);

/**
 * @brief Finds the bounding boxes containing the coins in the images
 * 
 * @param images vector of gray scale images
 * @param margin margin to add to the bounding boxes
 * 
 * @return vector of bounding boxes
 */
std::vector<cv::Rect> get_bbox_containing_coins(const std::vector<cv::Mat>& images, const int margin);

/**
 * @brief Cuts the images according to the given bounding boxes
 * 
 * @param images vector of gray scale images
 * @param cuts vector of bounding boxes
 * 
 * @return vector of cut images
 */
std::vector<cv::Mat> cut_images(const std::vector<cv::Mat>& images, const std::vector<cv::Rect>& cuts);

/**
 * @brief Applies preprocessing to a vector of images
 * 
 * @param images Vector of input images
 * @param param points points of the intersections of the piece wise linear transformation
 * @param s size of the Gaussian kernel
 * @param sigma standard deviation for GaussianBlur
 * 
 * @return Vector of preprocessed images
 */
std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, const std::vector<cv::Point2f>& points, int s, float sigma);


// ----- TEMPLATE MATCHING -----

struct DetectedCoin {
    cv::Point center;   // Centro dell'oggetto
    float radius;       // Raggio dell'oggetto
    float confidence;   // Confidenza della rilevazione
    std::string class_name; // Classe dell'oggetto
};

/**
 * @brief Finds the positions of circles in the image
*
 * @param I input image in HSV format
 * @param downsampling_factor factor by which the image was downsampled
 * 
 * @return vector of circle positions (x, y, radius)
 */
std::vector<cv::Vec3f> get_circles_positions(const cv::Mat& I, const float downsampling_factor);

/**
 * @brief Splits the image into separate coin images based on detected circles
 *
 * @param I input image
 * @param circles vector of detected circle positions (x, y, radius)
 * @param margin margin to add around each coin image
 *
 * @return vector of coin images
 */
std::vector<cv::Mat> split_image_by_coins(const cv::Mat& I, const std::vector<cv::Vec3f>& circles, int margin);

/**
 * @brief Finds all the non overlapping matches which have values above a certain threshold
 *
 * @param result result matrix (output of matchTemplate)
 * @param threshold threshold value
 * @param template_size size of the template used for matching
 * @param label class of the template
 *
 * @return Vector of (center, radius, confidence, class) structs
 */
std::vector<DetectedCoin> get_positions_and_values_above_threshold(const cv::Mat& result, double threshold, double template_size, std::string label);

/**
 * @brief Finds the best match above a certain threshold
 *
 * @param result result matrix (output of matchTemplate)
 * @param threshold threshold value
 * @param template_size size of the template used for matching
 * @param label class of the template
 *
 * @return The best match (center, radius, confidence, class) struct
 */
DetectedCoin get_best_match_above_threshold(const cv::Mat& result, double threshold, double template_size, std::string label);

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
bool add_near_point(const DetectedCoin& new_point, std::vector<DetectedCoin>& points, double min_distance);

/**
 * @brief Creates rotated versions of a template image
 *
 * @param templ template image to rotate
 * @param num_rotations number of rotations
 * 
 * @return Vector of rotated images
 */
std::vector<cv::Mat> rotate_template(const cv::Mat& templ, const int num_rotations);


// ----- PERFORMANCE METRICS -----

/**
 * @brief Gets the labels from a folder containing text files
 *
 * @param folder_path path to the folder containing the label files
 * @param downsampling_factor factor by which the image was downsampled
 *
 * @return vector of vectors containing the detected coins for each image
 */
std::vector<std::vector<DetectedCoin>> get_labels_from_folder(const std::string& folder_path, const float downsampling_factor);

/**
 * @brief Computes the intersection over union between two detected coins
 *
 * @param label ground truth label
 * @param prediction predicted label
 *
 * @return IoU value
 */
float intersection_over_union(const DetectedCoin label, const DetectedCoin prediction);

/**
 * @brief Computes the mean intersection over union between ground truth and predicted labels
 *
 * @param ground_truth_labels vector of ground truth labels
 * @param predicted_labels vector of predicted labels
 *
 * @return mean IoU value
 */
float compute_mIoU(const std::vector<DetectedCoin> ground_truth_labels, const std::vector<DetectedCoin> predicted_labels);

#endif // MAIN_H
