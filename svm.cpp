#include "main.h"

cv::Mat getHOGFeatures(const cv::Mat& image)
{
    std::vector<float> hog_features;
    cv::Mat resized;
    // Parametri HOG
    cv::HOGDescriptor hog(
        cv::Size(256, 256), // winSize (uguale a resize)
        cv::Size(8, 8),   // blockSize
        cv::Size(4, 4),     // blockStride
        cv::Size(4, 4),     // cellSize
        9                    // nbins
    );
    cv::resize(image, resized, cv::Size(256, 256)); // dimensione fissa
    hog.compute(resized, hog_features);
    return cv::Mat(hog_features).clone().reshape(1,1);
}

void datasetWithFeatures(std::vector<cv::Mat> images, cv::Mat& features, cv::Mat& labels, int label) {
    for (cv::Mat& image : images) {
        cv::Mat hog_feature = getHOGFeatures(image);
        features.push_back(hog_feature);
        labels.push_back(label);
    }
}
