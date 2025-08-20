
#include "main.h"

//MODIFIED

cv::Mat correct_illumination(cv::Mat I){
    cv::Mat illumination, corrected;
    // apply a Gaussian filter with a large kernel to estimate the illumination
    cv::GaussianBlur(I, illumination, cv::Size(101, 101), 0);
    // convert the images to float for division
    I.convertTo(I, CV_32F);
    illumination.convertTo(illumination, CV_32F);
    cv::Scalar meanVal = cv::mean(illumination);
    // divide the original image by the illumination
    // and multiply by the mean value of the illumination (normalization)
    cv::divide(I, illumination, corrected);
    corrected *= cv::mean(illumination)[0];
    corrected.convertTo(corrected, CV_8U);

    return corrected;
}


cv::Mat contrast_stretching(cv::Mat I, int threshold) {
    // mask contains pixel of value <= threshold
    cv::Mat mask;
    cv::threshold(I, mask, threshold, 255, cv::THRESH_BINARY_INV);

    // Multiply pixel by L/threshold
    cv::Mat temp;
    I.convertTo(temp, CV_32F);
    temp = temp * 255/threshold;
    temp.convertTo(temp, CV_8U);

    // Apply the mask
    cv::Mat stretched;
    stretched = cv::Mat::ones(I.size(), I.type()) * 255;       // all pixel to 255
    temp.copyTo(stretched, mask);                              // overwrite pixel < threshold

    return stretched;
}

cv::Mat display_hist(cv::Mat I, int bins, std::string name) {
    cv::Mat hist;
    if (I.channels() > 1) {
        return hist;
    }

    int histSize[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    int channels[] = { 0 };

    int width = bins*2;
    int height = 400;

    cv::calcHist(&I, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    cv::Mat histImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(hist, hist, 0, height, cv::NORM_MINMAX);
    for (int i = 1; i < bins; i++) {
        cv::line(histImage,
                 cv::Point(2*(i-1), height - cvRound(hist.at<float>(i - 1))),
                 cv::Point(2*i, height - cvRound(hist.at<float>(i))),
                 cv::Scalar(255, 255, 255));
    }

    cv::namedWindow(name);
    cv::imshow(name, histImage);

    return hist;
}

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
std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, float T, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {

        cv::Mat new_image = contrast_stretching(img, T);

        cv::GaussianBlur(new_image, new_image, cv::Size(s, s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}