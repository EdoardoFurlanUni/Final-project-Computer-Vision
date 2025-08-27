#include "main.h"

std::vector<cv::Mat> images;
int slider = 22;
int maxSlider = 255;


// void on_trackbar(int pos, void* userdata) {
    // std::vector<cv::Mat> concatenated_images;
    // int ref_width = 700;
    // int ref_height = 700;

    // aggiorna tutti i parametri delle trackbar
    // slider = cv::getTrackbarPos("slider", "process");

    // for (cv::Mat image : images) {
    //     cv::Mat processed_image = image.clone();
    //     cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2HSV);

    //     // processa
    //     auto start = std::chrono::high_resolution_clock::now();

    //     // Apply a threshold on the saturation
    //     cv::Mat mask;
    //     cv::inRange(processed_image, cv::Scalar(0, 40, 0), cv::Scalar(180, 255, 255), mask);

    //     // Convert to grayscale and blur
    //     cv::Mat gray;
    //     if (mask.channels() == 3) {
    //         cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    //     } else {
    //         gray = mask.clone();
    //     }
    //     cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    //     // Find circles using Hough Transform
    //     std::vector<cv::Vec3f> circles;
    //     cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
    //                     1,     // dp
    //                     110,   // minDist
    //                     100, 30, // param1, param2
    //                     95, 210); // minRadius, maxRadius

    //     std::cout << "Found " << circles.size() << " circles in the image." << std::endl;

    //     // Sort circles by radius in ascending order
    //     std::sort(circles.begin(), circles.end(),
    //             [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });

    //     // Print radius of the circles
    //     for (size_t i = 0; i < circles.size(); i++) {
    //         std::cout << "Circle " << i << ": radius = " << circles[i][2] << std::endl;
    //     }

    //     // Draw circles on the original image
    //     cv::cvtColor(processed_image, processed_image, cv::COLOR_HSV2BGR);
    //     for (const auto& c : circles) {
    //         cv::Point center(cvRound(c[0]), cvRound(c[1]));
    //         int radius = cvRound(c[2]);

    //         cv::circle(processed_image, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
    //         cv::circle(processed_image, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
    //     }

    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> elapsed = end - start;
    //     std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    //     // resize images for concatenation
    //     cv::resize(image, image, cv::Size(ref_width, ref_height));
    //     cv::resize(processed_image, processed_image, cv::Size(ref_width, ref_height));

    //     // concatenate horizontally
    //     cv::Mat concatenatedH;
    //     cv::hconcat(image, processed_image, concatenatedH);
    //     concatenated_images.push_back(concatenatedH);
    // }

    // // concatenate vertically
    // cv::Mat concatenatedV;
    // cv::vconcat(concatenated_images, concatenatedV);
    // cv::imshow("process", concatenatedV);
// }

int main(int argc, const char* argv[]) {

    std::vector<std::string> filenames = {"../test/images/IMG_24.jpg"};
    cv::namedWindow("process", cv::WINDOW_KEEPRATIO);

    // Load images
    for (const auto& filename : filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    // // Create trackbars
    // cv::createTrackbar("slider", "process", NULL, maxSlider, on_trackbar);

    // // Initial call to display image
    // on_trackbar(0, 0);

    // cv::waitKey(0);

    // std::vector<cv::Mat> concatenated_images;
    // int ref_width = 700;
    // int ref_height = 700;

    for (cv::Mat image : images) {
        cv::Mat processed_image = image.clone();
        cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2HSV);

        // processa
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Vec3f> circles = get_circles_positions(processed_image);

        cv::cvtColor(processed_image, processed_image, cv::COLOR_HSV2BGR);

        std::cout << "Found " << circles.size() << " circles in the image." << std::endl;

        // Print radius of the circles
        for (size_t i = 0; i < circles.size(); i++) {
            std::cout << "Circle " << i << ": radius = " << circles[i][2] << std::endl;
        }

        // Draw circles on the original image
        for (const auto& c : circles) {
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);

            cv::circle(processed_image, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
            cv::circle(processed_image, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
        }

        std::vector<cv::Mat> coin_images = split_image_by_coins(processed_image, circles, 25);

        for (size_t i = 0; i < coin_images.size(); i++) {
            cv::imshow("Coin " + std::to_string(i), coin_images[i]);
            cv::waitKey(0);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        // // resize images for concatenation
        // cv::resize(image, image, cv::Size(ref_width, ref_height));
        // cv::resize(processed_image, processed_image, cv::Size(ref_width, ref_height));

        // // concatenate horizontally
        // cv::Mat concatenatedH;
        // cv::hconcat(image, processed_image, concatenatedH);
        // concatenated_images.push_back(concatenatedH);
    }

    // // concatenate vertically
    // cv::Mat concatenatedV;
    // cv::vconcat(concatenated_images, concatenatedV);
    // cv::imshow("process", concatenatedV);

    // cv::waitKey(0);


    return 0;
}
