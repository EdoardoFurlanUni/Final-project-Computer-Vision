#include "main.h"

cv::Mat image;
cv::Mat processed_image;
int lowThreshold = 100;
int highThreshold = 160;
const int maxThreshold = 300;

void on_trackbar(int pos, void* userdata) {
    cv::Mat canny_image;

    lowThreshold = cv::getTrackbarPos("Low Threshold", "process");
    highThreshold = cv::getTrackbarPos("High Threshold", "process");

    if (lowThreshold > highThreshold) {
        cv::setTrackbarPos("Low Threshold", "process", highThreshold);
        lowThreshold = highThreshold;
    }
    cv::Canny(processed_image, canny_image, lowThreshold, highThreshold);
    cv::Mat concatenated;
    std::vector<cv::Mat> images = {image, processed_image, canny_image};
    cv::hconcat(images, concatenated);
    cv::imshow("process", concatenated);
}

int main(int argc, const char* argv[])
{    
    const std::vector<std::string> filenames = {"../template/images/10_CENT/IMG_22_temp.jpg", "../test/images/IMG_24.jpg"};
    for (const std::string& filename : filenames) {
        image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        processed_image = image.clone();

        // std::vector<cv::Mat> rotated_images = rotate_template(image, 8);
        // cv::namedWindow("template", cv::WINDOW_KEEPRATIO);

        // for (size_t i = 0; i < rotated_images.size(); i++) {
        //     cv::imshow("template", rotated_images[i]);
        //     cv::waitKey(0);
        // }

        // processed_image = contrast_stretching(image, 0.8*255);
        cv::GaussianBlur(processed_image, processed_image, cv::Size(5,5), 1.5);
        processed_image = correct_illumination(processed_image);

        cv::namedWindow("process", cv::WINDOW_KEEPRATIO);

        // // se vuoi plottare il kernel gaussiano
        // cv::Mat gaussNorm;
        // cv::normalize(cv::getGaussianKernel(101, 0) * cv::getGaussianKernel(101, 0).t(), gaussNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
        // std::cout << cv::getGaussianKernel(5, 0) << std::endl;
        // cv::resize(gaussNorm, gaussNorm, cv::Size(), 500, 500, cv::INTER_NEAREST);
        // cv::imshow("2D Gaussian Kernel", gaussNorm);
        
        // Create trackbars for Canny thresholds
        cv::createTrackbar("High Threshold", "process", NULL, maxThreshold, on_trackbar);
        cv::createTrackbar("Low Threshold", "process", NULL, maxThreshold, on_trackbar);

        cv::setTrackbarPos("High Threshold", "process", highThreshold);
        cv::setTrackbarPos("Low Threshold", "process", lowThreshold);

        // Initial call to display image
        on_trackbar(0, 0);

        cv::waitKey(0);
    }

    return 0;
}
