#include "main.h"

std::vector<cv::Mat> images;
int slider = 2;
int maxSlider = 100;
// int tileGridSize = 8;
// int maxTileGridSize = 30;


void on_trackbar(int pos, void* userdata) {
    std::vector<cv::Mat> concatenated_images;
    int ref_width = 700;
    int ref_height = 700;

    // aggiorna tutti i parametri delle trackbar
    slider = cv::getTrackbarPos("slider", "process");
    // tileGridSize = cv::getTrackbarPos("Tile Grid Size", "process");

    for (cv::Mat image : images) {
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

        // resize images for concatenation
        cv::resize(image, image, cv::Size(ref_width, ref_height));
        cv::resize(processed_image, processed_image, cv::Size(ref_width, ref_height));

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

    std::vector<std::string> filenames = {"../template/images/10_CENT/IMG_22_temp.jpg", "../test/images/IMG_24.jpg"};
    cv::namedWindow("process", cv::WINDOW_KEEPRATIO);

    // Load images
    for (const auto& filename : filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    // Cut images
    std::vector<cv::Rect> cuts = get_bbox_containing_coins(images, 50);
    images = cut_images(images, cuts);
    
    // Create trackbars
    cv::createTrackbar("slider", "process", NULL, maxSlider, on_trackbar);

    // set initial position
    // cv::setTrackbarPos("slider", "process", slider);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    return 0;
}
