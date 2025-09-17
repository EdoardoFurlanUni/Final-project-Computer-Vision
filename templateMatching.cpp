//main author - Edoardo Furlan
#include "main.h"

std::vector<cv::Vec3f> get_circles_positions(const cv::Mat& I, const float downsampling_factor) {
    std::vector<cv::Vec3f> circles;

    // ----- first processing to find edges based on local variance -----
    cv::Mat I_gray_f;
    cv::cvtColor(I, I_gray_f, cv::COLOR_BGR2GRAY);
    I_gray_f.convertTo(I_gray_f, CV_32F);
    cv::Mat I_gray_sq = I_gray_f.mul(I_gray_f); // I^2

    cv::Mat mean, mean_sq;
    boxFilter(I_gray_f, mean, -1, cv::Size(15,15)); // E[x]
    boxFilter(I_gray_sq, mean_sq, -1, cv::Size(15,15)); // E[x^2]
    cv::Mat var = mean_sq - mean.mul(mean); // Var(x) = E[x^2] - (E[x])^2

    cv::Mat var_norm;
    cv::normalize(var, var_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat edges;
    cv::threshold(var_norm, edges, 30, 255, cv::THRESH_BINARY);

    // // uncomment to see the local variance *****
    // cv::namedWindow("Local variance", cv::WINDOW_KEEPRATIO);
    // cv::namedWindow("High local variance (probable coins)", cv::WINDOW_KEEPRATIO);
    // cv::imshow("Local variance", var_norm);
    // cv::imshow("High local variance (probable coins)", edges);

    // Find circles using Hough Transform
    cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT,
                    1.25,     // dp
                    200*downsampling_factor,   // minDist
                    100, 30, // param1, param2
                    95*downsampling_factor, 210*downsampling_factor); // minRadius, maxRadius

    // ----- second processing to select circles based on color contained -----
    cv::Mat I_lab;
    cv::cvtColor(I, I_lab, cv::COLOR_BGR2Lab);

    // Split in channels
    cv::Mat copper, gold, mask_lab;
    std::vector<cv::Mat> lab_planes(3);
    cv::split(I_lab, lab_planes);
    cv::Mat L = lab_planes[0];  // Lightness
    cv::Mat a = lab_planes[1];  // Green <-> Red
    cv::Mat b = lab_planes[2];  // Blue <-> Yellow

    // set a threshold in Lab format
    cv::inRange(a, 135, 255, copper);
    cv::inRange(b, 134, 255, gold);
    cv::bitwise_or(copper, gold, mask_lab);

    // // uncomment to see the copper *****
    // cv::namedWindow("Mask copper", cv::WINDOW_KEEPRATIO);
    // cv::imshow("Mask copper", copper);
    // // uncomment to see the gold *****
    // cv::namedWindow("Mask gold", cv::WINDOW_KEEPRATIO);
    // cv::imshow("Mask gold", gold);
    // // uncomment to see the mask_lab *****
    // cv::namedWindow("Mask lab", cv::WINDOW_KEEPRATIO);
    // cv::imshow("Mask lab", mask_lab);

    for (const auto& circle : circles) {
        // compute the average color inside the circle
        cv::Mat c = cv::Mat::zeros(mask_lab.size(), CV_8U);
        cv::circle(c, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(255), -1);
        cv::Scalar mean = cv::mean(mask_lab, c); // mean[0] contains the average value

        // std::cout << "Circle at (" << circle[0] << ", " << circle[1] << ") with radius " << circle[2] 
        //           << " has mean mask value: " << mean[0] << std::endl;
        // if the average color is above a certain threshold, keep the circle
        if (mean[0] < 100) { // threshold to be tuned
            circles.erase(std::remove(circles.begin(), circles.end(), circle), circles.end());
            // std::cout << "Circle rejected." << std::endl;
        }
    }

    // Sort circles by radius in ascending order
    std::sort(circles.begin(), circles.end(), [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });

    return circles;
}


std::vector<cv::Mat> split_image_by_coins(const cv::Mat& I, const std::vector<cv::Vec3f>& circles, int margin) {
    std::vector<cv::Mat> coin_images;
    for (const auto& circle : circles) {

        int x = static_cast<int>(circle[0]);
        int y = static_cast<int>(circle[1]);
        int r = static_cast<int>(circle[2]);

        // Define the bounding box for the coin
        int x_start = std::max(0, x - r - margin);
        int y_start = std::max(0, y - r - margin);
        int x_end = std::min(I.cols, x + r + margin);
        int y_end = std::min(I.rows, y + r + margin);

        cv::Rect roi(x_start, y_start, x_end - x_start, y_end - y_start);
        coin_images.push_back(I(roi).clone());

    }
    return coin_images;
}

DetectedCoin get_best_match_above_threshold(const cv::Mat& result, double threshold, double template_size, std::string label) {
    DetectedCoin best_match;
    best_match.confidence = -1; // initialization
    double radius = template_size / 2.0;

    // get best result
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    if (maxVal >= threshold) {
        best_match.center = maxLoc + cv::Point(radius, radius);
        best_match.radius = radius;
        best_match.class_name = label;
        best_match.confidence = maxVal;
    }

    return best_match;
}

std::vector<cv::Mat> rotate_template(const cv::Mat& templ, const int num_rotations) {
    std::vector<cv::Mat> rotated_templates;
    cv::Point2f center(templ.cols / 2.0f, templ.rows / 2.0f);

    for (int i = 0; i < num_rotations; ++i) {
        cv::Mat rotated;

        double angle = 360.0 * i / num_rotations;
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        
        cv::warpAffine(templ, rotated, rot_mat, templ.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
        rotated_templates.push_back(rotated);
    }
    return rotated_templates;
}

DetectedCoin detect_coin(const cv::Mat& coin_image, std::vector<cv::Mat>& templates, const std::string& coin_class, 
                         const int number_rotations, const float matching_threshold) {
    // best match with reference to the coin image
    DetectedCoin best_match;
    best_match.confidence = -1.0; // initialization

    // matching over all templates
    for (const cv::Mat& template_img : templates) {

        // if template is bigger than coin image, skip matching
        if (template_img.cols > coin_image.cols || template_img.rows > coin_image.rows) {
            // // Show skipped template *****
            // std::cout << "Skipping template of class " << coin_class << " of size " << template_img.size() << " for coin image of size " << coin_image.size() << std::endl;
            continue;
        }

        // rotate template
        std::vector<cv::Mat> rotations = rotate_template(template_img, number_rotations);
        for (const cv::Mat& rotated_template : rotations) {

            cv::Mat result;
            cv::matchTemplate(coin_image, rotated_template, result, cv::TM_CCOEFF_NORMED);

            DetectedCoin current_match = get_best_match_above_threshold(result, matching_threshold, template_img.cols, coin_class);

            if (current_match.confidence > best_match.confidence) {
                best_match = current_match;
            }
        }
    }
    

    return best_match;
}

std::vector<DetectedCoin> detect_all_coins(const std::vector<cv::Mat>& preprocessed_coin_images,
                                           const std::vector<std::string>& coins_classes, std::vector<std::vector<cv::Mat>>& preprocessed_dataset_images,
                                           const int number_rotations, const float matching_threshold, 
                                           const std::vector<cv::Vec3f>& circles_positions, const float coin_image_margin) {
    // list of detected coins with reference to the whole test image
    std::vector<DetectedCoin> detected_coins;   // center, radius, confidence, class

    // loop over all coins sub-images
    for (size_t j = 0; j < preprocessed_coin_images.size(); j++) {
        cv::Mat coin_img = preprocessed_coin_images[j];

        // best match with reference to the coin image
        DetectedCoin best_match;
        best_match.confidence = -1.0; // initialization

        // matching over all classes of templates
        for (size_t c = 0; c < coins_classes.size(); c++) {

            DetectedCoin current_match = detect_coin(coin_img, preprocessed_dataset_images[c], coins_classes[c], number_rotations, matching_threshold);

            if (current_match.confidence > best_match.confidence) {
                best_match = current_match;
            }
        }

        // if a match was found convert it from coin reference frame to test image frame
        if (best_match.confidence > 0) {

            best_match.center += cv::Point(std::max(0.0f, circles_positions[j][0] - circles_positions[j][2] - coin_image_margin), std::max(0.0f, circles_positions[j][1] - circles_positions[j][2] - coin_image_margin));

            detected_coins.push_back(best_match);
        }
    }

    return detected_coins;
}
