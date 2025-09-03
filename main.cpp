#include "main.h"

int main(int argc, const char* argv[])
{
    const std::vector<std::string> coins_classes = {
        "EUR_001", "EUR_002", "EUR_010", "EUR_020", "EUR_050", "EUR_100", "EUR_200"
    };

    const std::vector<std::string> dataset_images_paths= {
        "../template/images/EUR_001", "../template/images/EUR_002", "../template/images/EUR_010", "../template/images/EUR_020", "../template/images/EUR_050", "../template/images/EUR_100", "../template/images/EUR_200"
    };
    const std::vector<std::string> dataset_labels_paths= {
        "../dataset/labels/EUR_001", "../dataset/labels/EUR_002", "../dataset/labels/EUR_010", "../dataset/labels/EUR_020", "../dataset/labels/EUR_050", "../dataset/labels/EUR_100", "../dataset/labels/EUR_200"
    };

    const std::string test_images_path = "../test/images/";
    const std::string test_labels_path = "../test/labels/";
    const std::string test_videos_labels_path_1 = "../test/videos/video1_frame/labels/";
    const std::string test_videos_labels_path_2 = "../test/videos/video2_frame/labels/";
    const std::vector<std::string> test_videos_path = {
        "../test/videos/", "../test/videos/video1_frame/images", "../test/videos/video2_frame/images"
    };


    double mIoUB = {0}; // the mean intersection of union computed over all the coins
    std::array<double, 8> accuracy = {0}; // an object is considered to be recognized only if its IoU is greater than 50%
    double sum_accuracy = 0; // average distance by the true sum in the images


    // ----- LOAD IMAGES -----
    const float downsampling_factor = 0.75;

    // load images in dataset path
    std::vector<std::vector<cv::Mat>> dataset_images_gray;
    dataset_images_gray.reserve(dataset_images_paths.size());
    for (const std::string& folder : dataset_images_paths) {

        std::vector<cv::Mat> images_in_folder = load_images_from_folder(folder, cv::IMREAD_GRAYSCALE);
        
        // downsample
        for (cv::Mat& image : images_in_folder) {
            cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
        }

        dataset_images_gray.push_back(images_in_folder);
    }

    // load images in test path
    /*std::vector<cv::Mat> test_images_gray = load_images_from_folder(test_images_path, cv::IMREAD_GRAYSCALE);
    for (cv::Mat& image : test_images_gray) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }
    std::vector<cv::Mat> test_images_colour = load_images_from_folder(test_images_path, cv::IMREAD_COLOR);
    for (cv::Mat& image : test_images_colour) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }*/

    // ----- PREPROCESSING (dataset and test) -----
    const int coin_image_margin = static_cast<int>(25*downsampling_factor);
    const std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
    const int gaussian_kernel_size = 3;
    const float gaussian_kernel_sigma = 1;

    // preprocess dataset images
    std::vector<std::vector<cv::Rect>> cuts_dataset_images;
    std::vector<std::vector<cv::Mat>> preprocessed_dataset_images;
    preprocessed_dataset_images.reserve(dataset_images_gray.size());
    for (const auto& imgs_in_folder : dataset_images_gray) {

        std::vector<cv::Mat> prep_imgs_in_folder = preprocess_images(imgs_in_folder, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);
        preprocessed_dataset_images.push_back(prep_imgs_in_folder);
    }
    // // to show preprocessed images *****
    // for (const auto& imgs_in_folder : preprocessed_dataset_images) {
    //     for (const cv::Mat& img : imgs_in_folder) {
    //         cv::imshow("Preprocessed Image", img);
    //         cv::waitKey(0);
    //     }
    // }

    // compute circles locations and split gray-scale test images in smaller images containing coins
    // then preprocess those images
    cv::VideoCapture cap("../test/videos/video1.MOV"); //modify the video to open
    int video = 1; // choose which video to use (1 or 2)
    if (!cap.isOpened()) {
        std::cerr << "Errore: impossibile aprire il video!" << std::endl;
        return -1;
    }

    cv::Mat original_frame, frame, last;
    cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);

    std::vector<DetectedCoin> last_coins_found;
    std::vector<cv::Vec3f> last_circles_found;
    std::vector<std::vector<DetectedCoin>> ground_truth_labels;
    std::vector<std::vector<DetectedCoin>> predicted_labels;
    std::vector<std::vector<cv::Vec3f>> predicted_circles;


    const int number_of_images = 5;
    int frame_indexes[number_of_images];

    if (video == 1){
        ground_truth_labels = get_labels_from_folder(test_videos_labels_path_1, 1.89f*downsampling_factor);
        frame_indexes[0] = 156;
        frame_indexes[1] = 250;
        frame_indexes[2] = 377;
        frame_indexes[3] = 452;
        frame_indexes[4] = 562;
    } else {
        frame_indexes[0] = 117;
        frame_indexes[1] = 185;
        frame_indexes[2] = 251;
        frame_indexes[3] = 332;
        frame_indexes[4] = 532;
        ground_truth_labels = get_labels_from_folder(test_videos_labels_path_2, 1.89f*downsampling_factor);
    }
    float sum = 0;
    int frame_count = 0;
    std::vector<cv::Mat> test_images_colour; // array that will contain the frames to be labeled for the performance metrics

    while (true) {
        
        cap >> original_frame;
        frame_count++;
        std::cout << "Processing frame: " << frame_count << std::endl;
        if (original_frame.empty()) break;  // fine del video
    
        // upscale and downsample to obtain the same size for coins
        cv::resize(original_frame, original_frame, cv::Size(), downsampling_factor*1.89f, downsampling_factor*1.89f);
        frame = original_frame.clone();
        // CLAHE to improve contrast and avoid illumination problems
        cv::Mat lab;
        cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_planes;
        cv::split(lab, lab_planes);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4.0);          // limits on contrast
        clahe->setTilesGridSize(cv::Size(16,16)); // size of the grid for histogram equalization
        clahe->apply(lab_planes[0], lab_planes[0]);

        cv::merge(lab_planes, lab);
        cv::cvtColor(lab, frame, cv::COLOR_Lab2BGR);
        
        // OPZIONALE -> per tornare indietro commentare e a riga 125 sostituire con maxDiff < 0.92 || maxDiff > 0.94
        cv::resize(frame, frame, cv::Size(), 0.5f, 0.5f);

        //cv::namedWindow("CLAHE", cv::WINDOW_KEEPRATIO);
        //cv::imshow("CLAHE", frame);
        //cv::waitKey(0);
        // ***TOLGIERE QUESTO IF SE NON SI VUOLE IL CONTROLLO DI SIMILARITA' TRA FRAME***
        if(!last.empty()) {
            double maxDiff;
            // compute difference between frames
            maxDiff = getSSIM(frame, last);
            std::cout << "Max difference between frames: " << maxDiff << std::endl;
            // if similarity is in this range, use last coins found
            if (maxDiff < 0.9 || maxDiff > 0.92) {          
                sum = 0;      
                for (const auto& d : last_coins_found) {
                    cv::circle(original_frame, d.center, d.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
                    cv::putText(original_frame, d.class_name, cv::Point(d.center.x, d.center.y - 10), cv::FONT_HERSHEY_SIMPLEX, 2*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));
                    sum += std::stof(d.class_name.substr(4)) / 100.0f; // extract last 3 char from class_name and cast to float
                }
                std::ostringstream out; // to not overwrite cout
                out << std::fixed << std::setprecision(2);  // fix at 2 decimals
                out << sum;
                std::string sum_str = out.str();

                cv::putText(original_frame, "Sum: " + sum_str + " euro", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 3 * downsampling_factor, cv::Scalar(0, 0, 255), static_cast<int>(5 * downsampling_factor));
                cv::imshow("Template Matching", original_frame);
                // show current frame with last coins found
                cv::imshow("Template Matching", original_frame);
                        //add labels found to predicted_circles
                for (int idx : frame_indexes) {
                    if (frame_count == idx) {
                        // Store predicted labels for each image
                        predicted_labels.push_back(last_coins_found);
                        test_images_colour.push_back(original_frame.clone());
                        predicted_circles.push_back(last_circles_found);
                        for (const auto& d : last_coins_found) {
                            std::cout << d.class_name << " ";
                        }
                    }
                }
                last = frame.clone();
                char c = (char)cv::waitKey(1);
                if (c == 27) break;
                continue;
            }

        }
        last = frame.clone();

        // reset frame
        frame = original_frame.clone();

        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Vec3f> circles = get_circles_positions(frame, downsampling_factor);

        std::vector<cv::Mat> coin_images = split_image_by_coins(frame_gray, circles, coin_image_margin);
        std::vector<cv::Mat> preprocessed_coin_images = preprocess_images(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

        // // Print number of circles found *****
        // std::cout << "Found " << preprocessed_test_images_coins.size() << " circles in the image." << std::endl;
        //cv::waitKey(0);
        // Print radius of the circles *****
        // for (size_t j = 0; j < circles.size(); j++) {
        //      std::cout << "Circle " << j << ": radius = " << circles[j][2] << std::endl;
        // }

        // Draw circles on the original image *****
        for (size_t j = 0; j < circles.size(); j++) {
            cv::Vec3f circle = circles[j];

            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int radius = cvRound(circle[2]);

            cv::circle(original_frame, center, radius, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor), cv::LINE_AA);
            cv::putText(original_frame, std::to_string(j), center, cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor));

            // // Show circles on original image *****
            // cv::namedWindow("Hough Circles", cv::WINDOW_KEEPRATIO);
            // cv::imshow("Hough Circles", test_images_colour[i]);
            // cv::waitKey(0);

            // // Show coin images *****
            //cv::namedWindow("Coin", cv::WINDOW_KEEPRATIO);
            //cv::imshow("Coin", preprocessed_coin_images[j]);
            //cv::waitKey(0);

            // // Show size of coin image *****
            // cv::Size coin_size = preprocessed_coin_images[j].size();
            // std::cout << "Coin " << j << " size: " << coin_size.width << "x" << coin_size.height << std::endl;
        }    


        // ----- TEMPLATE MATCHING (test) -----

        std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class

        auto start = std::chrono::high_resolution_clock::now();

        // loop over all coins sub-images
        for (size_t j = 0; j < preprocessed_coin_images.size(); j++) {
            cv::Mat coin_img = preprocessed_coin_images[j];

            // best match with reference to the coin image
            DetectedCoin best_match;
            best_match.confidence = -1.0; // initialization
            // matching over all templates
            for (size_t c = 0; c < coins_classes.size(); c++) {
                for (const cv::Mat& template_img : preprocessed_dataset_images[c]) {

                    // if template is bigger than coin image, skip matching
                    if (template_img.cols > coin_img.cols || template_img.rows > coin_img.rows) {
                        // // Show skipped template *****
                        // std::cout << "Skipping template of class " << coins_classes[c] << " of size " << template_img.size() << " for coin image " << j << " of size " << coin_img.size() << std::endl;
                        continue;
                    }

                    // rotate template
                    std::vector<cv::Mat> rotations = rotate_template(template_img, 8);
                    for (const cv::Mat& rotated_template : rotations) {

                        cv::Mat result;

                        cv::matchTemplate(coin_img, rotated_template, result, cv::TM_CCORR_NORMED);

                        DetectedCoin current_match = get_best_match_above_threshold(result, 0.75, template_img.cols, coins_classes[c]);
                        // // Print current match confidence *****
                        // std::cout << "Current match confidence: " << current_match.confidence << std::endl;

                        if (current_match.confidence > best_match.confidence) {
                            best_match = current_match;
                        }
                    }
                }
            }
            // // Print best match confidence *****
            // std::cout << "Best match confidence: " << best_match.confidence << std::endl; 
            // if a match was found convert it from coin reference frame to test image frame
            if (best_match.confidence > 0) {

                // // Show circle found in coin image *****
                // cv::namedWindow("Coin", cv::WINDOW_KEEPRATIO); 
                // cv::cvtColor(coin_img, coin_img, cv::COLOR_GRAY2BGR);
                // cv::circle(coin_img, best_match.center, best_match.radius, cv::Scalar(0, 255, 0), 5);
                // cv::imshow("Coin", coin_img);
                // cv::waitKey(0);
                best_match.center += cv::Point(std::max(0.0f, circles[j][0] - circles[j][2] - coin_image_margin), std::max(0.0f, circles[j][1] - circles[j][2] - coin_image_margin));

                // // Show circle found in whole image *****
                // cv::namedWindow("whole image", cv::WINDOW_KEEPRATIO);
                // cv::circle(test_images_colour[i], best_match.center, best_match.radius, cv::Scalar(0, 255, 0), 5);
                // cv::imshow("whole image", test_images_colour[i]);
                // cv::waitKey(0);

                coins_found.push_back(best_match);
            }
        }
        //copy coins_found in last_coins_found
        last_coins_found = coins_found;
        last_circles_found = circles;
        //add labels found to predicted_circles
        for (int idx : frame_indexes) {
            if (frame_count == idx) {
                // Store predicted labels for each image
                predicted_labels.push_back(coins_found);
                test_images_colour.push_back(original_frame.clone());
                predicted_circles.push_back(circles);
                for (const auto& d : coins_found) {
                    std::cout << d.class_name << " ";
                }
            }
        }


        // Measure the time taken for template matching
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        //std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        // Show all labels on the test image *****
        sum = 0;
        for (const auto& d : coins_found) {
            cv::circle(original_frame, d.center, d.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
            cv::putText(original_frame, d.class_name, cv::Point(d.center.x, d.center.y - 10), cv::FONT_HERSHEY_SIMPLEX, 2*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));
            sum += std::stof(d.class_name.substr(4)) / 100.0f; // extract last 3 char from class_name and cast to float

        }
        std::ostringstream out; // to not overwrite cout
        out << std::fixed << std::setprecision(2);  // fix at 2 decimals
        out << sum;
        std::string sum_str = out.str();

        cv::putText(original_frame, "Sum: " + sum_str + " euro", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 3 * downsampling_factor, cv::Scalar(0, 0, 255), static_cast<int>(5 * downsampling_factor));
        cv::imshow("Template Matching", original_frame);
        std::cout << "number of matches: " << coins_found.size() << std::endl;
        // cv::waitKey(0);
        // Esc per uscire
        char c = (char)cv::waitKey(1);
        if (c == 27) break;

    }

    for (size_t i = 0; i < number_of_images; i++) {

        std::cout << "######### Results for image " << i << " #########" << std::endl;

        cv::Point2f score = compute_mIoU_and_accuracy(ground_truth_labels[i], predicted_labels[i]);
        std::cout << "mIoU: " << score.x << std::endl;
        std::cout << "Accuracy: " << score.y * 100 << "%" << std::endl;

        float true_sum = sum_coins(ground_truth_labels[i]);
        std::cout << "True sum of coins: " << true_sum << std::endl;
        float pred_sum = sum_coins(predicted_labels[i]);
        std::cout << "Predicted sum of coins: " << pred_sum << std::endl;
        float diff_sum = cv::abs(true_sum - pred_sum);
        std::ostringstream out; // to not overwrite cout
        out << std::fixed << std::setprecision(2);  // fix at 2 decimals
        out << "True sum of coins: " << true_sum << " Predicted sum of coins: " << pred_sum << " Difference: " << diff_sum << std::endl;
        std::cout << out.str(); 

        // Draw circles
        for (size_t j = 0; j < predicted_circles[i].size(); j++) {
            cv::Vec3f& c = predicted_circles[i][j];
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);

            cv::circle(test_images_colour[i], center, radius, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor), cv::LINE_AA);
            cv::putText(test_images_colour[i], std::to_string(j), cv::Point(center.x-radius/2, center.y+radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor));
        }
        // Draw ground truth coins
        for (const auto& coin: ground_truth_labels[i]) {
            cv::circle(test_images_colour[i], coin.center, coin.radius, cv::Scalar(255, 255, 255), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
            cv::putText(test_images_colour[i], coin.class_name, cv::Point(coin.center.x-coin.radius/2, coin.center.y-coin.radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 255, 255), static_cast<int>(5*downsampling_factor));
        }
        // Draw predicted coins
        for (const auto& coin: predicted_labels[i]) {
            cv::circle(test_images_colour[i], coin.center, coin.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
            cv::putText(test_images_colour[i], coin.class_name, cv::Point(coin.center.x-coin.radius/2, coin.center.y+coin.radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));
        }
        cv::namedWindow("Results", cv::WINDOW_KEEPRATIO);
        cv::imshow("Results", test_images_colour[i]);
        cv::waitKey(0);
    }
    return 0;
}

