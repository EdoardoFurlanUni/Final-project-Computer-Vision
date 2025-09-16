#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <string>
#include <tuple>
#include <filesystem>
#include <dirent.h> 
#include <fstream>
#include <chrono>   // time measurement
#include <iomanip> // used for std::setprecision
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using Detection = std::tuple<cv::Point, double>;


// ----- FILE MANAGEMENT -----

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
 * @brief Apply the transformation T(r) = {where x1 < r < x2 then r * (y2-y1)/(x2-x1)}
 * 
 * @param I gray scale image
 * @param points points of the intersections of the piece wise linear transformation (0,0) (100,100)... 
 * 
 * @return The resulting image after the transformation
 */
cv::Mat contrast_stretching(const cv::Mat& I, const std::vector<cv::Point2f>& points);

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

/**
 * @brief Computes the Structural Similarity Index (SSIM) between two images
 * 
 * @param img1 first image
 * @param img2 second image
 * 
 * @return SSIM value
 */
double getSSIM(const cv::Mat& img1, const cv::Mat& img2);


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
 * @brief Creates rotated versions of a template image
 *
 * @param templ template image to rotate
 * @param num_rotations number of rotations
 * 
 * @return Vector of rotated images
 */
std::vector<cv::Mat> rotate_template(const cv::Mat& templ, const int num_rotations);

/**
 * @brief Detects if there is a coin in the given image using template matching
 *
 * @param coin_image the image containing one circle
 * @param templates the set of templates to match against
 * @param coin_class the class label of the coin
 * @param number_rotations the number of rotations to apply to each template
 * @param matching_threshold the threshold for considering a match valid
 *
 * @return the best matching coin detection, if there is none matching then DetectedCoin.confidence = -1
 */
DetectedCoin detect_coin(const cv::Mat& coin_image, std::vector<cv::Mat>& templates, const std::string& coin_class, 
                         const int number_rotations, const float matching_threshold);

/**
 * @brief Detects all coins in the given splitted image using template matching
 *
 * @param preprocessed_coin_images the splitted images containing one possible coin each
 * @param coins_classes the classes of the coins
 * @param preprocessed_dataset_images the preprocessed dataset images 
 * @param number_rotations the number of rotations to apply to each template 
 * @param matching_threshold the threshold for considering a match valid
 * @param circles_positions the positions of the detected circles -> used for returning to total image frame
 * @param coin_image_margin the margin added around each coin image -> used for returning to total image frame
 *
 * @return a vector of detected coins for each image
 */
std::vector<DetectedCoin> detect_all_coins(const std::vector<cv::Mat>& preprocessed_coin_images,
                                           const std::vector<std::string>& coins_classes, std::vector<std::vector<cv::Mat>>& preprocessed_dataset_images,
                                           const int number_rotations, const float matching_threshold, 
                                           const std::vector<cv::Vec3f>& circles_positions, const float coin_image_margin);

// ----- PERFORMANCE METRICS -----

/**
 * @brief Displays a progress bar in the console
 *
 * @param current current progress
 * @param total total value for completion
 * @param bar_width width of the progress bar
 */
void progress_bar(int current, int total, int bar_width);

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
 * @brief Computes the intersection and union between two detected coins
 *
 * @param label ground truth label
 * @param prediction predicted label
 *
 * @return (intersection_area, union_area)
 */
cv::Point2f intersection_and_union(const DetectedCoin label, const DetectedCoin prediction);

/**
 * @brief Computes the mean intersection over union between ground truth and predicted labels and the accuracy score
 *
 * @param ground_truth_labels vector of ground truth labels
 * @param predicted_labels vector of predicted labels
 *
 * @return mean IoU value and accuracy
 */
cv::Point2f compute_mIoU_and_accuracy(const std::vector<DetectedCoin> ground_truth_labels, const std::vector<DetectedCoin> predicted_labels);

/**
 * @brief Sums the values of the coins
 *
 * @param coins vector of detected coins
 *
 * @return total value
 */
float sum_coins(const std::vector<DetectedCoin>& coins);

#endif // MAIN_H
