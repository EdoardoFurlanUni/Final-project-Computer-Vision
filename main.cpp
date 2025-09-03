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
    std::vector<cv::Mat> test_images_gray = load_images_from_folder(test_images_path, cv::IMREAD_GRAYSCALE);
    for (cv::Mat& image : test_images_gray) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }
    //std::vector<cv::Mat> test_images_colour = load_images_from_folder(test_images_path, cv::IMREAD_COLOR);
    //for (cv::Mat& image : test_images_colour) {
        //cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    //}

    // ----- PREPROCESSING (dataset) -----
    const std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
    const int gaussian_kernel_size = 3;
    const float gaussian_kernel_sigma = 1;

    // preprocess dataset images
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


    // // ----- TEMPLATE MATCHING (test images) -----
    // std::vector<std::vector<cv::Vec3f>> predicted_circles;
    // std::vector<std::vector<DetectedCoin>> predicted_labels;
    // const int number_of_images = static_cast<int>(test_images_colour.size());
    // const int coin_image_margin = static_cast<int>(25*downsampling_factor);
    // const int rotations = 8;
    // const float matching_threshold = 0.5;

    // // loop over all test images
    // for (size_t i = 0; i < number_of_images; i++) {

    //     // search for circles
    //     std::vector<cv::Vec3f> circles = get_circles_positions(test_images_colour[i], downsampling_factor);
    //     predicted_circles.push_back(circles);

    //     // split and preprocess the whole image in sub-images containing coins
    //     std::vector<cv::Mat> coin_images = split_image_by_coins(test_images_gray[i], circles, coin_image_margin);
    //     std::vector<cv::Mat> preprocessed_coin_images = preprocess_images(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

    //     auto start = std::chrono::high_resolution_clock::now();

    //     // list of detected coins with reference to the whole test image
    //     std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class
    //     coins_found = detect_all_coins(preprocessed_coin_images, coins_classes, preprocessed_dataset_images, rotations, matching_threshold, circles, coin_image_margin);

    //     // Store predicted labels for each image
    //     predicted_labels.push_back(coins_found);

    //     // Measure the time taken for template matching
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> elapsed = end - start;
    //     std::cout << "Image " << i << " / " << number_of_images-1 << " : " << elapsed.count() << " seconds" << std::endl;
    // }


    // // ----- PERFORMANCE METRICS AND RESULTS (test images) -----
    // std::vector<std::vector<DetectedCoin>> ground_truth_labels = get_labels_from_folder(test_labels_path, downsampling_factor);

    // for (size_t i = 0; i < number_of_images; i++) {

    //     std::cout << "######### Results for image " << i << " #########" << std::endl;

    //     cv::Point2f score = compute_mIoU_and_accuracy(ground_truth_labels[i], predicted_labels[i]);
    //     std::cout << "mIoU: " << score.x << std::endl;
    //     std::cout << "Accuracy: " << score.y * 100 << "%" << std::endl;

    //     float true_sum = sum_coins(ground_truth_labels[i]);
    //     float pred_sum = sum_coins(predicted_labels[i]);
    //     float diff_sum = cv::abs(true_sum - pred_sum);
    //     std::ostringstream out; // to not overwrite cout
    //     out << std::fixed << std::setprecision(2);  // fix at 2 decimals
    //     out << "True sum of coins: " << true_sum << " Predicted sum of coins: " << pred_sum << " Difference: " << diff_sum << std::endl;
    //     std::cout << out.str(); 

    //     // Draw circles
    //     for (size_t j = 0; j < predicted_circles[i].size(); j++) {
    //         cv::Vec3f& c = predicted_circles[i][j];
    //         cv::Point center(cvRound(c[0]), cvRound(c[1]));
    //         int radius = cvRound(c[2]);

    //         cv::circle(test_images_colour[i], center, radius, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor), cv::LINE_AA);
    //         cv::putText(test_images_colour[i], std::to_string(j), cv::Point(center.x-radius/2, center.y+radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor));
    //     }
    //     // Draw ground truth coins
    //     for (const auto& coin: ground_truth_labels[i]) {
    //         cv::circle(test_images_colour[i], coin.center, coin.radius, cv::Scalar(255, 255, 255), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
    //         cv::putText(test_images_colour[i], coin.class_name, cv::Point(coin.center.x-coin.radius/2, coin.center.y-coin.radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 255, 255), static_cast<int>(5*downsampling_factor));
    //     }
    //     // Draw predicted coins
    //     for (const auto& coin: predicted_labels[i]) {
    //         cv::circle(test_images_colour[i], coin.center, coin.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
    //         cv::putText(test_images_colour[i], coin.class_name, cv::Point(coin.center.x-coin.radius/2, coin.center.y+coin.radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));
    //     }
    //     cv::namedWindow("Results", cv::WINDOW_KEEPRATIO);
    //     cv::imshow("Results", test_images_colour[i]);
    //     cv::waitKey(0);
    // }
    // // close all windows
    // cv::destroyAllWindows();


    // ----- TEMPLATE MATCHING (test videos) -----

    cv::VideoCapture cap("../test/videos/video1.MOV");
    int video = 1;
    if (!cap.isOpened()) {
        std::cerr << "Errore: impossibile aprire il video!" << std::endl;
        return -1;
    }

    int fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(cvRound(0.5f*downsampling_factor*1.89f*cap.get(cv::CAP_PROP_FRAME_WIDTH)), cvRound(0.5f*downsampling_factor*1.89f*cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer("../test/videos/output.MOV",
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps,
                           frameSize,
                           true); // true = color video

    cv::Mat original_frame, frame, last;
    std::vector<DetectedCoin> last_coins_found;
    std::vector<std::vector<DetectedCoin>> predicted_labels; // only of the frames for compute performances (156 250 377 452 562) TODO!
    std::vector<cv::Vec3f> last_circles_found;
    std::vector<std::vector<DetectedCoin>> ground_truth_labels;
    std::vector<std::vector<cv::Vec3f>> predicted_circles;
    int similarity_timer = 0; // every 3 similar analyze the frame

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

    cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);

    while (true) {
        cap >> original_frame;
        frame_count++;
        if (original_frame.empty()) break;

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
        cv::resize(frame, frame, cv::Size(), 0.25f, 0.25f);

        if(!last.empty()) {
            // compute similarity between frames
            double similarity = getSSIM(frame, last);
            const float different_threshold = 0.83f;
            const float similar_threshold = 0.845f;

            if (similarity_timer >= 3) { // every 3 similar analyze the frame
                similarity_timer = 0;
                similarity = similar_threshold - 0.001f; // to force analysis
            }

            std::cout << "Similarity between frames: " << similarity;
            if (similarity < different_threshold) {
                similarity_timer = 0;
                std::cout << " different" << std::endl;
            }
            else if (similarity > similar_threshold) {
                similarity_timer++;
                std::cout << " similar" << std::endl;
            }
            else {
                similarity_timer = 0;
                std::cout << " uncertain" << std::endl;
            }
            // if similarity is in this range, use last coins found
            if (similarity < different_threshold || similarity > similar_threshold) {
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
        
                // save frame for output video
                cv::resize(original_frame, original_frame, cv::Size(), 0.5, 0.5); // rescale to reduce video dimension
                writer.write(original_frame);
                // show current frame with last coins found
                cv::imshow("Template Matching", original_frame);

                for (int idx : frame_indexes) {
                    if (frame_count == idx) {
                        // Store predicted labels for each image
                        predicted_labels.push_back(last_coins_found);
                        test_images_colour.push_back(original_frame.clone());
                        predicted_circles.push_back(last_circles_found);
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

        const int coin_image_margin = static_cast<int>(25*downsampling_factor);
        const int rotations = 8;
        const float matching_threshold = 0.5;

        // search for circles
        std::vector<cv::Vec3f> circles = get_circles_positions(frame, downsampling_factor);

        // split and preprocess the whole image in sub-images containing coins
        std::vector<cv::Mat> coin_images = split_image_by_coins(frame_gray, circles, coin_image_margin);
        std::vector<cv::Mat> preprocessed_coin_images = preprocess_images(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

        std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class
        coins_found = detect_all_coins(preprocessed_coin_images, coins_classes, preprocessed_dataset_images, rotations, matching_threshold, circles, coin_image_margin);
        last_coins_found = coins_found;
        last_circles_found = circles;

        //add labels found to predicted_circles
        for (int idx : frame_indexes) {
            if (frame_count == idx) {
                // Store predicted labels for each image
                predicted_labels.push_back(coins_found);
                test_images_colour.push_back(original_frame.clone());
                predicted_circles.push_back(circles);
            }
        }

        // Draw circles
        for (size_t j = 0; j < circles.size(); j++) {
            cv::Vec3f& c = circles[j];
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);

            cv::circle(original_frame, center, radius, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor), cv::LINE_AA);
            cv::putText(original_frame, std::to_string(j), cv::Point(center.x-radius/2, center.y+radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor));

        }
        // Draw predicted coins
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
       
        // save frame for output video
        cv::resize(original_frame, original_frame, cv::Size(), 0.5, 0.5); // rescale to reduce video dimension
        writer.write(original_frame);

        cv::imshow("Template Matching", original_frame);
        // Esc per uscire
        char c = (char)cv::waitKey(1);
        if (c == 27) break;

    }

    cap.release();
    writer.release();

    cv::VideoCapture capOut("../test/videos/output.MOV");
    if (!capOut.isOpened()) {
        std::cerr << "Errore apertura video output!" << std::endl;
        return -1;
    }

    while (true) {
        capOut >> frame;
        if (frame.empty()) break;

        cv::namedWindow("Processed Video", cv::WINDOW_KEEPRATIO);
        cv::imshow("Processed Video", frame);
        if (cv::waitKey(1000 / fps) == 27) break; // ESC per uscire
    }

    capOut.release();
    cv::destroyAllWindows();

    // ----- PERFORMANCE METRICS AND RESULTS (test videos) -----
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

