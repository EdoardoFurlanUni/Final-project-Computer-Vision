#include "main.h"

int main(int argc, const char* argv[])
{
    // ----- PATHS -----
    const std::vector<std::string> coins_classes = {
        "EUR_001", "EUR_002", "EUR_010", "EUR_020", "EUR_050", "EUR_100", "EUR_200"
    };

    const std::vector<std::string> dataset_images_paths= {
        "../template/images/EUR_001/*.jpg", "../template/images/EUR_002/*.jpg", "../template/images/EUR_010/*.jpg", "../template/images/EUR_020/*.jpg", "../template/images/EUR_050/*.jpg", "../template/images/EUR_100/*.jpg", "../template/images/EUR_200/*.jpg"
    };
    const std::vector<std::string> dataset_labels_paths= {
        "../dataset/labels/EUR_001/*.jpg", "../dataset/labels/EUR_002/*.jpg", "../dataset/labels/EUR_010/*.jpg", "../dataset/labels/EUR_020/*.jpg", "../dataset/labels/EUR_050/*.jpg", "../dataset/labels/EUR_100/*.jpg", "../dataset/labels/EUR_200/*.jpg"
    };

    const std::string test_images_path = "../test/images/*.jpg";
    const std::string test_labels_path = "../test/labels/*.txt";
    const std::string test_videos_labels_path_1 = "../test/videos/video1_frame/labels/*.txt";
    const std::string test_videos_labels_path_2 = "../test/videos/video2_frame/labels/*.txt";
    const std::vector<std::string> test_videos_path = {
        "../test/videos/", "../test/videos/video1_frame/images", "../test/videos/video2_frame/images"
    };

    std::string results_path;


    // ----- LOAD IMAGES (dataset) -----
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

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video1|video2|images>" << std::endl;
        return 1;
    }


    // ----- DETECTION (test images or videos) -----
    std::vector<std::vector<cv::Vec3f>> predicted_circles;
    std::vector<std::vector<DetectedCoin>> predicted_labels;
    std::vector<std::vector<DetectedCoin>> ground_truth_labels;
    std::vector<cv::Mat> test_images_colour;

    const int coin_image_margin = static_cast<int>(25*downsampling_factor);
    const int rotations = 8;
    const float matching_threshold = 0.5;

    // division of detection between images and videos
    if(std::string(argv[1]) == "images") {
        std::vector<cv::Mat> test_images_gray;
        results_path = "../results/images";
        // load images in test path
        test_images_gray = load_images_from_folder(test_images_path, cv::IMREAD_GRAYSCALE);
        for (cv::Mat& image : test_images_gray) {
            cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
        }
        test_images_colour = load_images_from_folder(test_images_path, cv::IMREAD_COLOR);
        for (cv::Mat& image : test_images_colour) {
            cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
        }
        // load ground truth labels for performance evaluation
        ground_truth_labels = get_labels_from_folder(test_labels_path, downsampling_factor);


        // ----- TEMPLATE MATCHING (test images) -----

        // loop over all test images
        for (size_t i = 0; i < test_images_colour.size(); i++) {

            // search for circles
            std::vector<cv::Vec3f> circles = get_circles_positions(test_images_colour[i], downsampling_factor);
            predicted_circles.push_back(circles);

            // split and preprocess the whole image in sub-images containing coins
            std::vector<cv::Mat> coin_images = split_image_by_coins(test_images_gray[i], circles, coin_image_margin);
            std::vector<cv::Mat> preprocessed_coin_images = preprocess_images(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

            auto start = std::chrono::high_resolution_clock::now();

            // list of detected coins with reference to the whole test image
            std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class
            coins_found = detect_all_coins(preprocessed_coin_images, coins_classes, preprocessed_dataset_images, rotations, matching_threshold, circles, coin_image_margin);

            // Store predicted labels for each image
            predicted_labels.push_back(coins_found);

            // Measure the time taken for template matching
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Image " << i << " / " << test_images_colour.size()-1 << " : " << elapsed.count() << " seconds" << std::endl;
        }
    }
    else if (std::string(argv[1]) == "video1" || std::string(argv[1]) == "video2"){
        cv::Mat original_frame, frame, last;
        std::vector<DetectedCoin> last_coins_found;
        std::vector<cv::Vec3f> last_circles_found;

        // open input video file
        cv::VideoCapture cap("../test/videos/" + std::string(argv[1]) + ".MOV");
        if (!cap.isOpened()) {
            std::cerr << "Error: can't open video!" << std::endl;
            return -1;
        }

        // define the video writer
        int fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(cvRound(0.3f*downsampling_factor*1.89f*cap.get(cv::CAP_PROP_FRAME_WIDTH)), cvRound(0.3f*downsampling_factor*1.89f*cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        results_path = "../results/" + std::string(argv[1]);
        // Crea la cartella se non esiste
        if (!std::filesystem::exists(results_path)) {
            if (!std::filesystem::create_directories(results_path)) {
                std::cerr << "Error: can't open folder:  " << results_path << std::endl;
                return 0;
            }
        }
        
        std::string output_path = results_path + "/output_" + std::string(argv[1]) + ".avi";
        cv::VideoWriter writer(output_path,
                            cv::VideoWriter::fourcc('M','J','P','G'),
                            fps,
                            frameSize,
                            true); // true = color video

        // load ground truth labels for performance evaluation
        std::vector<int> frame_indexes;
        if (std::string(argv[1]) == "video1"){
            ground_truth_labels = get_labels_from_folder(test_videos_labels_path_1, 1.89f*downsampling_factor);
            frame_indexes.push_back(156);
            frame_indexes.push_back(250);
            frame_indexes.push_back(377);
            frame_indexes.push_back(452);
            frame_indexes.push_back(562);
        } else {
            ground_truth_labels = get_labels_from_folder(test_videos_labels_path_2, 1.89f*downsampling_factor);
            frame_indexes.push_back(117);
            frame_indexes.push_back(185);
            frame_indexes.push_back(251);
            frame_indexes.push_back(332);
            frame_indexes.push_back(532);
        }

        // ----- TEMPLATE MATCHING (test videos) -----
        cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);
        std::cout << "Press ESC to exit" << std::endl;

        const int max_similar_frames = 10;
        int similarity_timer = max_similar_frames; // to force analysis on first frames
        const float different_threshold = 0.83f; // thresholds to decide if frames are different
        const float similar_threshold = 0.845f; // or similar
        float sum = 0;
        int frame_count = 0;
        
        while (true) {
            cap >> original_frame;

            if (original_frame.empty()) break;

            // show progress bar
            frame_count++;
            progress_bar(frame_count, static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)), 100);
            
            // if (frame_count < 100) {
            //     continue;
            // }

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

            cv::resize(frame, frame, cv::Size(), 0.25f, 0.25f);

            if(!last.empty()) {
                // compute similarity between frames
                double similarity = getSSIM(frame, last);

                // check if similarity timer is expired or if current frame is in the list of frames to be analyzed
                if (similarity_timer == 0 || std::find(frame_indexes.begin(), frame_indexes.end(), frame_count) != frame_indexes.end()) {
                    similarity_timer = max_similar_frames;
                    similarity = similar_threshold - 0.001f; // to force analysis
                }

                if (similarity < different_threshold) {
                    similarity_timer = max_similar_frames;
                }
                else if (similarity > similar_threshold) {
                    similarity_timer--;
                }
                else {
                    similarity_timer = max_similar_frames;
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
                    out << std::fixed << std::setprecision(2) << sum;  // fix at 2 decimals
                    cv::putText(original_frame, "Sum: " + out.str() + " euro", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 3 * downsampling_factor, cv::Scalar(0, 0, 255), static_cast<int>(5 * downsampling_factor));
            
                    // save frame for output video
                    cv::resize(original_frame, original_frame, cv::Size(), 0.3, 0.3); // rescale to reduce video dimension
                    writer.write(original_frame);

                    // show current frame with last coins found
                    cv::imshow("Template Matching", original_frame);

                    if (std::find(frame_indexes.begin(), frame_indexes.end(), frame_count) != frame_indexes.end()) {
                        // Store predicted labels for each image
                        test_images_colour.push_back(original_frame.clone());
                        predicted_circles.push_back(last_circles_found);
                        predicted_labels.push_back(last_coins_found);
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

            // search for circles
            std::vector<cv::Vec3f> circles = get_circles_positions(frame, downsampling_factor);

            // split and preprocess the whole image in sub-images containing coins
            std::vector<cv::Mat> coin_images = split_image_by_coins(frame_gray, circles, coin_image_margin);
            std::vector<cv::Mat> preprocessed_coin_images = preprocess_images(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);

            std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class
            coins_found = detect_all_coins(preprocessed_coin_images, coins_classes, preprocessed_dataset_images, rotations, matching_threshold, circles, coin_image_margin);
            last_coins_found = coins_found;
            last_circles_found = circles;

            if (std::find(frame_indexes.begin(), frame_indexes.end(), frame_count) != frame_indexes.end()) {
                // Store predicted labels for each image
                test_images_colour.push_back(original_frame.clone());
                predicted_circles.push_back(circles);
                predicted_labels.push_back(coins_found);
            }
            // Draw predicted coins
            sum = 0;
            for (const auto& d : coins_found) {
                cv::circle(original_frame, d.center, d.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
                cv::putText(original_frame, d.class_name, cv::Point(d.center.x, d.center.y - 10), cv::FONT_HERSHEY_SIMPLEX, 2*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));
                sum += std::stof(d.class_name.substr(4)) / 100.0f; // extract last 3 char from class_name and cast to float
            }
            std::ostringstream out; // to not overwrite cout
            out << std::fixed << std::setprecision(2) << sum;  // fix at 2 decimal;
            cv::putText(original_frame, "Sum: " + out.str() + " euro", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 3 * downsampling_factor, cv::Scalar(0, 0, 255), static_cast<int>(5 * downsampling_factor));
        
            // save frame for output video
            cv::resize(original_frame, original_frame, cv::Size(), 0.3, 0.3); // rescale to reduce video dimension
            writer.write(original_frame);

            cv::imshow("Template Matching", original_frame);
            // Esc per uscire
            char c = (char)cv::waitKey(1);
            if (c == 27) break;

        }

        cap.release();
        writer.release();

        cv::VideoCapture capOut(output_path);
        if (!capOut.isOpened()) {
            std::cerr << "Error opening output video!" << std::endl;
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
    }
    else{
        std::cerr << "Non valid input: " << std::string(argv[1]) << ". Use video1, video2 or images." << std::endl;
        return 1;
    }

    // ----- PERFORMANCE METRICS AND RESULTS (test videos) -----
    for (size_t i = 0; i < test_images_colour.size(); i++) {

        std::cout << "######### Results for image " << i << " #########" << std::endl;

        cv::Point2f score = compute_mIoU_and_accuracy(ground_truth_labels[i], predicted_labels[i]);
        std::cout << "mIoU: " << score.x << std::endl;
        std::cout << "Accuracy: " << score.y * 100 << "%" << std::endl;

        float true_sum = sum_coins(ground_truth_labels[i]);
        float pred_sum = sum_coins(predicted_labels[i]);
        float diff_sum = cv::abs(true_sum - pred_sum);
        std::ostringstream out; // to not overwrite cout
        out << std::fixed << std::setprecision(2);  // fix at 2 decimals
        out << "True sum of coins: " << true_sum << ", predicted sum of coins: " << pred_sum << ", difference: " << diff_sum << std::endl;
        std::cout << out.str();

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



        std::cout << "Press any key to continue..." << std::endl;
        cv::waitKey(0);
    }


    // Crea la cartella se non esiste
    if (!std::filesystem::exists(results_path)) {
        if (!std::filesystem::create_directories(results_path)) {
            std::cerr << "Error: can't open folder:  " << results_path << std::endl;
            return 0;
        }
    }

    for (size_t i = 0; i < test_images_colour.size(); ++i) {
        std::ostringstream oss;
        oss << results_path << "/" << "frame_" << i << ".jpg";
        const std::string filename = oss.str();

        if (!cv::imwrite(filename, test_images_colour[i])) {
            std::cerr << "Error in saving image: " << filename << std::endl;
        }
    }
    return 0;
}

