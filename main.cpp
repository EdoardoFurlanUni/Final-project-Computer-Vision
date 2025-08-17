#include "main.h"

int main(int argc, const char* argv[])
{
    const std::array<std::string, 8> coins_classes = {
        {"1_CENT", "2_CENT", "5_CENT", "10_CENT", "20_CENT", "50_CENT", "1_EURO", "2_EURO"}
    };

    const std::array<std::string, 8> dataset_images_paths= {
        {"./dataset/images/1_CENT", "./dataset/images/2_CENT", "./dataset/images/5_CENT", "./dataset/images/10_CENT", "./dataset/images/20_CENT", "./dataset/images/50_CENT", "./dataset/images/1_EURO", "./dataset/images/2_EURO"}
    };
    const std::array<std::string, 8> dataset_labels_paths= {
        {"./dataset/labels/1_CENT", "./dataset/labels/2_CENT", "./dataset/labels/5_CENT", "./dataset/labels/10_CENT", "./dataset/labels/20_CENT", "./dataset/labels/50_CENT", "./dataset/labels/1_EURO", "./dataset/labels/2_EURO"}
    };

    const std::string test_images_path = "./test/images/";
    const std::string test_labels_path = "./test/labels/";

    const std::array<std::string, 3> test_videos_path = {
        {"./test/videos/", "./test/videos/video1_frame/images", "./test/videos/video2_frame/images"}
    };


    double mIoUB = {0}; // the mean intersection of union computed over all the coins
    std::array<double, 8> accuracy = {0}; // an object is considered to be recognized only if its IoU is greater than 50%
    double sum_accuracy = 0; // average distance by the true sum in the images

    // ----- PREPROCESSING (dataset and test) -----

    // ----- HOUGH CIRCLES (dataset and test) -----

    // ----- SIFT OR TEMPLATE MATCHING (test) -----

    // ----- COMPUTE OUTPUT (test) -----

    // ----- PERFORMANCE METRICS (test) -----

    std::cout << "si" << std::endl;
    
    return 0;
}

