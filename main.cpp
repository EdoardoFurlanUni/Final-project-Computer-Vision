#include "main.h"

int main(int argc, const char* argv[])
{
    const std::vector<std::string> coins_classes = {
        "1_CENT", "2_CENT", "10_CENT", "20_CENT", "50_CENT", "1_EURO", "2_EURO"
    };

    const std::vector<std::string> dataset_images_paths= {
        "../template/images/1_CENT", "../template/images/2_CENT", "../template/images/10_CENT", "../template/images/20_CENT", "../template/images/50_CENT", "../template/images/1_EURO", "../template/images/2_EURO"
    };
    const std::vector<std::string> dataset_labels_paths= {
        "../dataset/labels/1_CENT", "../dataset/labels/2_CENT", "../dataset/labels/10_CENT", "../dataset/labels/20_CENT", "../dataset/labels/50_CENT", "../dataset/labels/1_EURO", "../dataset/labels/2_EURO"
    };

    const std::string test_images_path = "../test/images/";
    const std::string test_labels_path = "../test/labels/";

    const std::vector<std::string> test_videos_path = {
        "../test/videos/", "../test/videos/video1_frame/images", "../test/videos/video2_frame/images"
    };


    double mIoUB = {0}; // the mean intersection of union computed over all the coins
    std::array<double, 8> accuracy = {0}; // an object is considered to be recognized only if its IoU is greater than 50%
    double sum_accuracy = 0; // average distance by the true sum in the images


    // ----- LOAD IMAGES -----
    const float downsampling_factor = 1;

    // load images in dataset path
    std::vector<std::vector<cv::Mat>> dataset_images_gray;
    dataset_images_gray.reserve(dataset_images_paths.size());
    size_t coin_class = 1;
    cv::Mat trainingData, trainingLabels;
    for (const std::string& folder : dataset_images_paths) {

        std::vector<cv::Mat> images_in_folder = load_images_from_folder(folder, cv::IMREAD_GRAYSCALE);

        datasetWithFeatures(images_in_folder, trainingData, trainingLabels, coin_class);
        // downsample
        for (cv::Mat& image : images_in_folder) {
            cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
        }

        dataset_images_gray.push_back(images_in_folder);
        coin_class++;
    }

    // Converti in float per la SVM
    trainingData.convertTo(trainingData, CV_32F);

    // Crea e configura la SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR); // puoi provare anche RBF
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));


    // Grid search su C e gamma
    cv::ml::ParamGrid Cgrid   = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C);
    cv::ml::ParamGrid gammaGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA);
    cv::ml::ParamGrid pGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P);
    cv::ml::ParamGrid nuGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU);
    cv::ml::ParamGrid coeffGrid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF);

        // Allena con trainAuto → fa cross-validation e cerca i parametri migliori
    svm->trainAuto(cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, trainingLabels),
                   10, // k-fold cross-validation
                   Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, cv::ml::ParamGrid());


    std::cout << "Best C: " << svm->getC() << std::endl;
    std::cout << "Best Gamma: " << svm->getGamma() << std::endl;
    std::cout << "Best Kernel: " << svm->getKernelType() << std::endl;

     // Salva il modello
    svm->save("coin_svm.yml");
    std::cout << "Modello salvato in coin_svm.yml" << std::endl;
    // load images in test path
    std::vector<cv::Mat> test_images_gray = load_images_from_folder(test_images_path, cv::IMREAD_GRAYSCALE);
    for (cv::Mat& image : test_images_gray) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }
    std::vector<cv::Mat> test_images_colour = load_images_from_folder(test_images_path, cv::IMREAD_COLOR);
    for (cv::Mat& image : test_images_colour) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }

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

    // compute circles locations and split gray-scale test images in smaller images containig coins
    // then preprocess those images
    std::vector<std::vector<cv::Vec3f>> circles_positions;
    std::vector<std::vector<cv::Mat>> preprocessed_test_images_coins;
    preprocessed_test_images_coins.reserve(test_images_gray.size());

    for (size_t i = 0; i < test_images_gray.size(); i++) {
        cv::Mat img_HSV;
        cv::cvtColor(test_images_colour[i], img_HSV, cv::COLOR_BGR2HSV);

        std::vector<cv::Vec3f> circles = get_circles_positions(img_HSV, downsampling_factor);
        circles_positions.push_back(circles);

        std::vector<cv::Mat> coin_images = split_image_by_coins(test_images_gray[i], circles, coin_image_margin);
        std::vector<cv::Mat> preprocessed_coin_images = preprocess_images_test(coin_images, points_contrast_stretching, gaussian_kernel_size, gaussian_kernel_sigma);
        preprocessed_test_images_coins.push_back(preprocessed_coin_images);

        // // Print number of circles found *****
        // std::cout << "Found " << circles.size() << " circles in the image." << std::endl;

        // // Print radius of the circles *****
        // for (size_t j = 0; j < circles.size(); j++) {
        //     std::cout << "Circle " << j << ": radius = " << circles[j][2] << std::endl;
        // }

        // Draw circles on the original image *****
        for (size_t j = 0; j < circles.size(); j++) {
            cv::Vec3f circle = circles[j];

            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int radius = cvRound(circle[2]);

            cv::circle(test_images_colour[i], center, radius, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor), cv::LINE_AA);
            cv::putText(test_images_colour[i], std::to_string(j), center, cv::FONT_HERSHEY_SIMPLEX, 1.5*downsampling_factor, cv::Scalar(255, 0, 255), static_cast<int>(3*downsampling_factor));

            // // Show circles on original image *****
            // cv::namedWindow("Hough Circles", cv::WINDOW_KEEPRATIO);
            // cv::imshow("Hough Circles", test_images_colour[i]);
            // cv::waitKey(0);

            // Show coin images *****
            // cv::namedWindow("Coin", cv::WINDOW_KEEPRATIO);
            // cv::imshow("Coin", preprocessed_coin_images[j]);
            // cv::waitKey(0);

            // // Show size of coin image *****
            // cv::Size coin_size = preprocessed_coin_images[j].size();
            // std::cout << "Coin " << j << " size: " << coin_size.width << "x" << coin_size.height << std::endl;
        }
    }    


    // ----- TEMPLATE MATCHING (test) -----
    cv::namedWindow("Template Matching", cv::WINDOW_KEEPRATIO);

    // loop over all test images
    for (size_t i = 0; i < preprocessed_test_images_coins.size(); i++) {
        const std::vector<cv::Mat>& coin_images = preprocessed_test_images_coins[i];

        auto start = std::chrono::high_resolution_clock::now();

        // list of detected coins with reference to the whole test image
        std::vector<DetectedCoin> coins_found;     // center, radius, confidence, class
        DetectedCoin match;
        // loop over all coins sub-images
        for (size_t j = 0; j < coin_images.size(); j++) {

            cv::Mat coin_img = coin_images[j];
            cv::namedWindow("Coin Image", cv::WINDOW_KEEPRATIO);
            cv::imshow("Coin Image", coin_img);
            cv::Mat features = getHOGFeatures(coin_img);
            features.convertTo(features, CV_32F);
            int response = (int)svm->predict(features);
            std::cout << "Classe predetta: " << response << std::endl;
            match.class_name = coins_classes[response - 1];
            match.confidence = 1.0; // SVM does not provide a confidence score
            match.radius = circles_positions[i][j][2];
            match.center = cv::Point(circles_positions[i][j][0], circles_positions[i][j][1]);
            cv::waitKey(0);

            // // Print best match confidence *****
            // std::cout << "Best match confidence: " << best_match.confidence << std::endl; 
            // if a match was found convert it from coin reference frame to test image frame

        // Draw all labels
        cv::circle(test_images_colour[i], match.center, match.radius, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor), cv::LINE_AA);
        cv::putText(test_images_colour[i], match.class_name, cv::Point(match.center.x, match.center.y - 10), cv::FONT_HERSHEY_SIMPLEX, 2*downsampling_factor, cv::Scalar(0, 255, 0), static_cast<int>(5*downsampling_factor));

        // Measure the time taken for template matching
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        cv::imshow("Template Matching", test_images_colour[i]);
        std::cout << "number of matches: " << coins_found.size() << std::endl;
        cv::waitKey(0);
    }
    }
    // ----- HOUGH LINES (test) -----
    // ----- COMPUTE OUTPUT (test) -----

    // ----- PERFORMANCE METRICS (test) -----
    
    return 0;
}


