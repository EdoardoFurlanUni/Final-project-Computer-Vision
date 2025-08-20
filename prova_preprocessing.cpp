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
    const std::string filename = "../test/images/IMG_24.jpg";
    image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    image = contrast_stretching(image, 0.97*255);
    cv::GaussianBlur(image, image, cv::Size(5,5), 1.5);
    image = correct_illumination(image);
    cv::namedWindow("test image", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("canny image", cv::WINDOW_KEEPRATIO);

    // se vuoi plottare il kernel gaussiano
    // cv::Mat gaussNorm;
    // cv::normalize(cv::getGaussianKernel(5, 1.5) * cv::getGaussianKernel(5, 1.5).t(), gaussNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::resize(gaussNorm, gaussNorm, cv::Size(), 20, 20, cv::INTER_NEAREST);
    // cv::imshow("2D Gaussian Kernel", gaussNorm);

    cv::imshow("test image", image);
    
    // Create trackbars for Canny thresholds
    cv::createTrackbar("High Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    cv::createTrackbar("Low Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    
    cv::setTrackbarPos("High Threshold", "canny image", highThreshold);
    cv::setTrackbarPos("Low Threshold", "canny image", lowThreshold);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    return 0;
}
