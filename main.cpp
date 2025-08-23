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
        {"../template/labels/1_CENT", "../template/labels/2_CENT", "../template/labels/10_CENT", "../template/labels/20_CENT", "../template/labels/50_CENT", "../template/labels/1_EURO", "../template/labels/2_EURO"}
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
    std::vector<std::vector<cv::Mat>> dataset_images;
    dataset_images.reserve(dataset_images_paths.size());
    for (const std::string& folder : dataset_images_paths) {

        std::vector<cv::Mat> images_in_folder = load_images_from_folder(folder);
        dataset_images.push_back(images_in_folder);
    }

    // load images in test path
    std::vector<cv::Mat> test_images = load_images_from_folder(test_images_path);

    // ----- PREPROCESSING (dataset and test) -----
    float contrast_stretching_T = 0.5*255; 
    int gaussian_kernel_size = 5;
    float gaussian_kernel_sigma = 1.5;
    
    // preprocess dataset images
    std::vector<std::vector<cv::Mat>> preprocessed_dataset_images;
    preprocessed_dataset_images.reserve(dataset_images.size());
    for (const auto& imgs_in_folder : dataset_images) {

        std::vector<cv::Mat> prep_imgs_in_folder = preprocess_images(imgs_in_folder, contrast_stretching_T, gaussian_kernel_size, gaussian_kernel_sigma);
        preprocessed_dataset_images.push_back(prep_imgs_in_folder);
    }
    // // to show preprocessed images
    for (const auto& imgs_in_folder : preprocessed_dataset_images) {
         for (const cv::Mat& img : imgs_in_folder) {
             cv::imshow("Preprocessed Image", img);
             cv::waitKey(0);
        }
    }

    // preprocess test images
    std::vector<cv::Mat> preprocessed_test_images = preprocess_images_test(test_images, contrast_stretching_T, gaussian_kernel_size, gaussian_kernel_sigma);

    // ----- HOUGH CIRCLES (dataset and test) -----
    /* for (const auto& imgs_in_folder : preprocessed_dataset_images) {
        for (const cv::Mat& img_1ch : imgs_in_folder) {
            cv::Mat img;
            cv::cvtColor(img_1ch, img, cv::COLOR_GRAY2BGR);
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 3, img.cols/8, 200, 100, 0, 0);
            std::cout << "Found " << circles.size() << " circles in the image." << std::endl;
            for( size_t i = 0; i < circles.size(); i++ )
            {
                cv::Vec3i c = circles[i];
                cv::Point center = cv::Point(c[0], c[1]);
                // circle center
                cv::circle(img, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
                // circle outline
                int radius = c[2];
                cv::circle(img, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
            }
            cv::namedWindow("Hough Circles");
            cv::imshow("Hough Circles", img);
            cv::waitKey(0);
        }
    }

    for (const cv::Mat& img : preprocessed_test_images) {
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(img, circles, cv::HOUGH_GRADIENT, 1, img.cols/8, 200, 100, 0, 0);
        std::cout << "Found " << circles.size() << " circles in the image." << std::endl;
        for( size_t i = 0; i < circles.size(); i++ )
        {
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);
            // circle center
            cv::circle(img, center, 1, cv::Scalar(0,0,100), 3, cv::LINE_AA);
            // circle outline
            int radius = c[2];
            cv::circle(img, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
        }
        cv::namedWindow("Hough Circles test");
        cv::imshow("Hough Circles test", img);
        cv::waitKey(0);*/
    //}
    // ----- TEMPLATE MATCHING (test) -----

    for (const cv::Mat& img : preprocessed_test_images) {
        auto start = std::chrono::high_resolution_clock::now();

        // convert to BGR to draw colored bbox
        cv::Mat img_3ch;
        cv::cvtColor(img, img_3ch, cv::COLOR_GRAY2BGR);

        std::vector<std::tuple<cv::Point, float>> positions_found;

        for (size_t i = 0; i < coins_classes.size(); i++) {
            for (const cv::Mat& template_img : preprocessed_dataset_images[i]) {
                
                // search for the best matches (at least > 0.95) among all the rotations of the template
                //std::vector<cv::Mat> rotations = rotate_template(template_img, 1);
                //for (const cv::Mat& rotated_template : rotations) {

                cv::Mat result;
                //Methods available for template matching: cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED
                cv::matchTemplate(img, template_img, result, cv::TM_CCOEFF_NORMED);
                //cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

                std::vector<std::tuple<cv::Point, float>> good_matches = get_positions_and_values_above_threshold(result, 0.7);

                for (const auto& match : good_matches) {
                    cv::Point loc = std::get<0>(match);
                    float val = std::get<1>(match);

                    cv::Point center = cv::Point(loc.x + template_img.cols/2, loc.y + template_img.rows/2);
                    float radius = template_img.rows/2;

                    // check if it has been already found
                    if (!exists_near_point(center, positions_found, radius, val)) {
                        for(const auto& d : positions_found) {
                            cv::Point p = std::get<0>(d);
                            float v = std::get<1>(d);
                            std::cout << "current point: " << p << " with confidence: " << v << std::endl;
                        }
                        cv::circle(img_3ch, center, radius, cv::Scalar(0, 255, 0), 5);
                        cv::putText(img_3ch, coins_classes[i], cv::Point(loc.x, loc.y - 10), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 255, 0), 5);
                        std::cout << "Found " << coins_classes[i] << " at location: " << loc << " with confidence: " << val << std::endl;
                    }
                }
                //}
            }
        }

        // Measure the time taken for template matching
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);
        cv::imshow("Template Matching", img_3ch);
        cv::waitKey(0);
    }

    // ----- HOUGH LINES (test) -----
    // ----- COMPUTE OUTPUT (test) -----

    // ----- PERFORMANCE METRICS (test) -----
    
    return 0;
}

