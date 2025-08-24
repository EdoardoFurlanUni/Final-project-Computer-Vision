#include "main.h"

std::vector<std::string> filenames;
int clipLimit = 2;
int maxClipLimit = 10;
// int tileGridSize = 8;
// int maxTileGridSize = 30;


void on_trackbar(int pos, void* userdata) {
    std::vector<cv::Mat> concatenated_images;
    int ref_width = 500;

    // aggiorna tutti i parametri delle trackbar
    clipLimit = cv::getTrackbarPos("Clip Limit", "process");
    // tileGridSize = cv::getTrackbarPos("Tile Grid Size", "process");

    for (const std::string& filename : filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        cv::Mat processed_image = image.clone();
        
        // processa
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
        processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(15, 15));
        clahe->apply(processed_image, processed_image);
        cv::GaussianBlur(processed_image, processed_image, cv::Size(3,3), 1.0);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        // std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
        // processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        // cv::Mat mask;
        // cv::threshold(processed_image, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU); // white coins
        // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        // cv::dilate(mask, mask, kernel);
        // cv::erode(mask, mask, kernel);
        // processed_image = mask;

        // std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.8*255, 255), cv::Point2f(255, 255)};
        // processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        // cv::GaussianBlur(processed_image, processed_image, cv::Size(5,5), sigma/100.0);

        // resize images for concatenation
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
    cv::createTrackbar("Clip Limit", "process", NULL, maxClipLimit, on_trackbar);
    // cv::createTrackbar("Tile Grid Size", "process", NULL, maxTileGridSize, on_trackbar);

    // set initial position
    cv::setTrackbarPos("Clip Limit", "process", clipLimit);
    // cv::setTrackbarPos("Tile Grid Size", "process", tileGridSize);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    return 0;
}
