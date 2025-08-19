#include "main.h"

int main(int argc, const char* argv[])
{
    const std::vector<std::string> coins_classes = {
        {"1_CENT", "2_CENT", "5_CENT", "10_CENT", "20_CENT", "50_CENT", "1_EURO", "2_EURO"}
    };

    const std::vector<std::string> dataset_images_paths= {
        {"./dataset/images/1_CENT", "./dataset/images/2_CENT", "./dataset/images/5_CENT", "./dataset/images/10_CENT", "./dataset/images/20_CENT", "./dataset/images/50_CENT", "./dataset/images/1_EURO", "./dataset/images/2_EURO"}
    };
    const std::vector<std::string> dataset_labels_paths= {
        {"./dataset/labels/1_CENT", "./dataset/labels/2_CENT", "./dataset/labels/5_CENT", "./dataset/labels/10_CENT", "./dataset/labels/20_CENT", "./dataset/labels/50_CENT", "./dataset/labels/1_EURO", "./dataset/labels/2_EURO"}
    };

    const std::string test_images_path = "./test/images/";
    const std::string test_labels_path = "./test/labels/";

    const std::vector<std::string> test_videos_path = {
        {"./test/videos/", "./test/videos/video1_frame/images", "./test/videos/video2_frame/images"}
    };


    double mIoUB = {0}; // the mean intersection of union computed over all the coins
    std::array<double, 8> accuracy = {0}; // an object is considered to be recognized only if its IoU is greater than 50%
    double sum_accuracy = 0; // average distance by the true sum in the images

    // ----- LOAD IMAGES -----
    std::vector<std::vector<cv::Mat>> dataset_images;
    dataset_images.reserve(dataset_images_paths.size());
    for (const std::string& folder : dataset_images_paths) {

        std::vector<cv::Mat> images_in_folder = load_images_from_folder(folder);
        dataset_images.push_back(images_in_folder);
    }

    std::vector<cv::Mat> test_images = load_images_from_folder(test_images_path);

    // ----- PREPROCESSING (dataset and test) -----
    float contrast_stretching_T = 0.4*255; 
    int gaussian_kernel_size = 5;
    float gaussian_kernel_sigma = 1.5;
    
    std::vector<std::vector<cv::Mat>> preprocessed_dataset_images;
    preprocessed_dataset_images.reserve(dataset_images.size());
    for (const auto& imgs_in_folder : dataset_images) {

        std::vector<cv::Mat> prep_imgs_in_folder = preprocess_images(imgs_in_folder, contrast_stretching_T, gaussian_kernel_size, gaussian_kernel_sigma);
        preprocessed_dataset_images.push_back(prep_imgs_in_folder);
    }

    std::vector<cv::Mat> preprocessed_test_images = preprocess_images(test_images, contrast_stretching_T, gaussian_kernel_size, gaussian_kernel_sigma);

    // ----- HOUGH CIRCLES (dataset and test) -----

    // ----- SIFT OR TEMPLATE MATCHING (test) -----

    // ----- COMPUTE OUTPUT (test) -----

    // ----- PERFORMANCE METRICS (test) -----

    std::cout << "si" << std::endl;
    
    return 0;
}

