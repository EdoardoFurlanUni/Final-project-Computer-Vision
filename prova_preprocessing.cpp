#include "main.h"

cv::Mat image;
int lowThreshold = 100;
int highThreshold = 160;
const int maxThreshold = 200;

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
    const std::string filename = "./dataset/images/1_CENT/IMG_18.jpg";
    image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::namedWindow("test image", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("canny image", cv::WINDOW_KEEPRATIO);

    image = contrast_streatching(image, 0.4*255);
    cv::blur(image, image, cv::Size(3,3));

    cv::imshow("test image", image);
    
    // Create trackbars for Canny thresholds
    cv::createTrackbar("High Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    cv::setTrackbarPos("High Threshold", "canny image", highThreshold);
    cv::createTrackbar("Low Threshold", "canny image", NULL, maxThreshold, on_trackbar);
    cv::setTrackbarPos("Low Threshold", "canny image", lowThreshold);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    return 0;
}
