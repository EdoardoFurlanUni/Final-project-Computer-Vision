#include "main.h"

cv::Mat image;
int lowThreshold = 100;
int highThreshold = 160;
const int maxThreshold = 300;

void on_trackbar(int pos, void* userdata) {
    cv::Mat canny_image;

    lowThreshold = cv::getTrackbarPos("Low Threshold", "canny image");
    highThreshold = cv::getTrackbarPos("High Threshold", "canny image");

    if (lowThreshold > highThreshold) {
        cv::setTrackbarPos("Low Threshold", "canny image", highThreshold);
        lowThreshold = highThreshold;
    }
    cv::Canny(image, canny_image, lowThreshold, highThreshold);
    cv::imshow("canny image", canny_image);
}

int main(int argc, const char* argv[])
{    
    const std::string filename = "../template/images/10_CENT/IMG_22_temp.jpg";
    image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    std::vector<cv::Mat> rotated_images = rotate_template(image, 8);
    cv::namedWindow("template", cv::WINDOW_KEEPRATIO);

    for (size_t i = 0; i < rotated_images.size(); i++) {
        cv::imshow("template", rotated_images[i]);
        cv::waitKey(0);
    }

    // image = contrast_stretching(image, 0.97*255);
    // cv::GaussianBlur(image, image, cv::Size(5,5), 1.5);
    // image = correct_illumination(image);
    // cv::namedWindow("test image", cv::WINDOW_KEEPRATIO);
    // cv::namedWindow("canny image", cv::WINDOW_KEEPRATIO);

    // // se vuoi plottare il kernel gaussiano
    // cv::Mat gaussNorm;
    // cv::normalize(cv::getGaussianKernel(101, 0) * cv::getGaussianKernel(101, 0).t(), gaussNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    // std::cout << cv::getGaussianKernel(5, 0) << std::endl;
    // cv::resize(gaussNorm, gaussNorm, cv::Size(), 500, 500, cv::INTER_NEAREST);
    // cv::imshow("2D Gaussian Kernel", gaussNorm);

    // cv::imshow("test image", image);
    
    // // Create trackbars for Canny thresholds
    // cv::createTrackbar("High Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    // cv::createTrackbar("Low Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    
    // cv::setTrackbarPos("High Threshold", "canny image", highThreshold);
    // cv::setTrackbarPos("Low Threshold", "canny image", lowThreshold);

    // // Initial call to display image
    // on_trackbar(0, 0);

    // cv::waitKey(0);

    return 0;
}
