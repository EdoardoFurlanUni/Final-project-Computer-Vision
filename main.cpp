#include "main.h"

int main(int argc, const char* argv[])
{
    const std::vector<std::string> coins_classes = {
        {"1_CENT", "2_CENT", "10_CENT", "20_CENT", "50_CENT", "1_EURO", "2_EURO"}
    };

    const std::vector<std::string> dataset_images_paths= {
        {"../template/images/1_CENT", "../template/images/2_CENT", "../template/images/10_CENT", "../template/images/20_CENT", "../template/images/50_CENT", "../template/images/1_EURO", "../template/images/2_EURO"}
    };
    const std::vector<std::string> dataset_labels_paths= {
        {"../dataset/labels/1_CENT", "../dataset/labels/2_CENT", "../dataset/labels/10_CENT", "../dataset/labels/20_CENT", "../dataset/labels/50_CENT", "../dataset/labels/1_EURO", "../dataset/labels/2_EURO"}
    };

    const std::string test_images_path = "../test/images/";
    const std::string test_labels_path = "../test/labels/";

    const std::vector<std::string> test_videos_path = {
        {"../test/videos/", "../test/videos/video1_frame/images", "../test/videos/video2_frame/images"}
    };


    double mIoUB = {0}; // the mean intersection of union computed over all the coins
    std::array<double, 8> accuracy = {0}; // an object is considered to be recognized only if its IoU is greater than 50%
    double sum_accuracy = 0; // average distance by the true sum in the images

    // ----- LOAD IMAGES -----
    // load images in dataset path
    std::vector<std::vector<cv::Mat>> dataset_images_gray;
    dataset_images_gray.reserve(dataset_images_paths.size());
    for (const std::string& folder : dataset_images_paths) {

        std::vector<cv::Mat> images_in_folder = load_images_from_folder(folder, cv::IMREAD_GRAYSCALE);
        dataset_images_gray.push_back(images_in_folder);
    }

    // load images in test path
    std::vector<cv::Mat> test_images_gray = load_images_from_folder(test_images_path, cv::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> test_images_colour = load_images_from_folder(test_images_path, cv::IMREAD_COLOR);

    // compute circles locations and split gray-scale test images containig coins
    std::vector<std::vector<cv::Vec3f>> circles_positions;
    std::vector<std::vector<cv::Mat>> test_images_coins;
    test_images_coins.reserve(test_images_gray.size());

    for (size_t i = 0; i < test_images_gray.size(); i++) {
        cv::Mat img_HSV;
        cv::cvtColor(test_images_colour[i], img_HSV, cv::COLOR_BGR2HSV);

        std::vector<cv::Vec3f> circles = get_circles_positions(img_HSV);
        circles_positions.push_back(circles);

        std::vector<cv::Mat> coin_images = split_image_by_coins(test_images_gray[i], circles, 25); // 25 is the margin to add
        test_images_coins.push_back(coin_images);

        for (const auto& coin : coin_images) {
            cv::namedWindow("Coin", cv::WINDOW_KEEPRATIO);
            cv::imshow("Coin", coin);
            cv::waitKey(0);
        }

        std::cout << "Found " << circles.size() << " circles in the image." << std::endl;

        // Print radius of the circles
        for (size_t j = 0; j < circles.size(); j++) {
            std::cout << "Circle " << j << ": radius = " << circles[j][2] << std::endl;
        }

        // // Draw circles on the original image
        // for (const auto& c : circles) {
        //     cv::Point center(cvRound(c[0]), cvRound(c[1]));
        //     int radius = cvRound(c[2]);

        //     cv::circle(test_images_colour[i], center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
        //     cv::circle(test_images_colour[i], center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
        // }
        // cv::namedWindow("Hough Circles", cv::WINDOW_KEEPRATIO);
        // cv::imshow("Hough Circles", test_images_colour[i]);
        // cv::waitKey(0);
    }

    // ----- PREPROCESSING (dataset and test) -----
    std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
    int gaussian_kernel_size = 3;
    float gaussian_kernel_sigma = 1;
    
    // cut and preprocess dataset images
    std::vector<std::vector<cv::Rect>> cuts_dataset_images;
    std::vector<std::vector<cv::Mat>> preprocessed_dataset_images;
    preprocessed_dataset_images.reserve(dataset_images_gray.size());
    for (const auto& imgs_in_folder : dataset_images_gray) {

        std::vector<cv::Rect> cuts = get_bbox_containing_coins(imgs_in_folder, 50);
        std::vector<cv::Mat> cut_imgs_in_folder = cut_images(imgs_in_folder, cuts);

        std::vector<cv::Mat> prep_imgs_in_folder = preprocess_images(cut_imgs_in_folder, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);
        preprocessed_dataset_images.push_back(prep_imgs_in_folder);
    }
    // to show preprocessed images
    for (const auto& imgs_in_folder : preprocessed_dataset_images) {
        for (const cv::Mat& img : imgs_in_folder) {
            cv::imshow("Preprocessed Image", img);
            cv::waitKey(0);
        }
    }

    // cut and preprocess test images
    std::vector<cv::Rect> cuts_test_images = get_bbox_containing_coins(test_images_gray, 50);
    std::vector<cv::Mat> preprocessed_test_images = preprocess_images_test(cut_images(test_images_gray, cuts_test_images), points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

    // segmentation of the test images
    for (const cv::Mat& img : preprocessed_test_images) {
        cv::imshow("Preprocessed Test Image", img);
        cv::waitKey(0);
    }
    
    // ----- TEMPLATE MATCHING (test) -----
    cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);

    for (const cv::Mat& img : preprocessed_test_images) {
        auto start = std::chrono::high_resolution_clock::now();

        // convert to BGR to draw colored bbox
        cv::Mat img_3ch;
        cv::cvtColor(img, img_3ch, cv::COLOR_GRAY2BGR);

        std::vector<DetectedCoin> positions_found;     // center, radius, confidence, class

        for (size_t i = 0; i < coins_classes.size(); i++) {
            for (const cv::Mat& template_img : preprocessed_dataset_images[i]) {

                std::vector<cv::Mat> rotations = rotate_template(template_img, 10);
                for (const cv::Mat& rotated_template : rotations) {

                    cv::Mat result;
                    //Methods available for template matching: cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED
                    cv::matchTemplate(img, rotated_template, result, cv::TM_CCOEFF_NORMED);

                    std::vector<DetectedCoin> good_matches = get_positions_and_values_above_threshold(result, 0.6, template_img.cols, coins_classes[i]);

                    for (const auto& match : good_matches) {

                        // check if it has been already found
                        if (!add_near_point(match, positions_found, match.radius)) {
                            for(const auto& d : positions_found) {
                                std::cout << "current point: " << d.center << " with confidence: " << d.confidence << std::endl;
                            }
                            std::cout << "Added " << coins_classes[i] << " at location: " << match.center << " with confidence: " << match.confidence << std::endl;
                    }
                }
            }
            }
        }

        // Draw all labels
        for (const auto& d : positions_found) {
            cv::circle(img_3ch, d.center, d.radius, cv::Scalar(0, 255, 0), 5);
            cv::putText(img_3ch, d.class_name, cv::Point(d.center.x, d.center.y - 10), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 255, 0), 5);
        }

        // Measure the time taken for template matching
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        cv::imshow("Template Matching", img_3ch);
        std::cout << "number of matches: " << positions_found.size() << std::endl;
        cv::waitKey(0);
    }

    // ----- HOUGH LINES (test) -----
    // ----- COMPUTE OUTPUT (test) -----

    // ----- PERFORMANCE METRICS (test) -----
    
    return 0;
}

