#include "main.h"

std::vector<std::string> filenames;
int sigma = 150;
int maxSigma = 300;


void on_trackbar(int pos, void* userdata) {
    std::vector<cv::Mat> concatenated_images;
    int ref_width = 500;

    // aggiorna tutti i parametri delle trackbar
    sigma = cv::getTrackbarPos("Sigma", "process");

    for (const std::string& filename : filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        cv::Mat processed_image = image.clone();
        
        // processa
        std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
        processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        cv::Mat mask;
        cv::threshold(processed_image, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU); // white coins
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(mask, mask, kernel);
        cv::erode(mask, mask, kernel);
        processed_image = mask;

        // std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.8*255, 255), cv::Point2f(255, 255)};
        // processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        // cv::GaussianBlur(processed_image, processed_image, cv::Size(5,5), sigma/100.0);

        // resize images
        cv::resize(image, image, cv::Size(ref_width, static_cast<int>(image.rows * ref_width / static_cast<float>(image.cols))));
        cv::resize(processed_image, processed_image, cv::Size(ref_width, static_cast<int>(processed_image.rows * ref_width / static_cast<float>(processed_image.cols))));

        // concatenate horizontally
        cv::Mat concatenatedH;
        cv::hconcat(image, processed_image, concatenatedH);
        concatenated_images.push_back(concatenatedH);
    }

    // concatenate vertically
    cv::Mat concatenatedV;
    cv::vconcat(concatenated_images, concatenatedV);
    cv::imshow("process", concatenatedV);
}

int main(int argc, const char* argv[]) {

    filenames = {"../template/images/10_CENT/IMG_22_temp.jpg", "../test/images/IMG_24.jpg"};
    cv::namedWindow("process", cv::WINDOW_KEEPRATIO);
        
    // Create trackbars
    cv::createTrackbar("Sigma", "process", NULL, maxSigma, on_trackbar);

    // set initial position
    cv::setTrackbarPos("Sigma", "process", sigma);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    return 0;
}
