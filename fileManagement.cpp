#include "main.h"

std::vector<cv::Mat> load_images_from_folder(const std::string& folder, int flags) {
    std::vector<cv::String> filenames;
    std::vector<cv::Mat> images;

    // Takes all images in the folder (jpg, png, ecc.)
    cv::glob(folder, filenames, false);

    // Load images in the order returned by glob (already lexicographically sorted)
    for (const auto& file : filenames) {
        cv::Mat img = cv::imread(file, flags);
        if (!img.empty()) {
            images.push_back(img);
        } else {
            std::cerr << "Impossible to load image: " << file << std::endl;
        }
    }
    return images;
}
