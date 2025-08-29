#include "main.h"
/*
std::vector<cv::Vec3f> get_circles_positions(const cv::Mat& I, const float downsampling_factor) {
    std::vector<cv::Vec3f> circles;

    // verify that I is in HSV format
    if (I.empty() || I.type() != CV_8UC3) {
        std::cerr << "Input image is empty or not in HSV format." << std::endl;
        return circles;
    }

    // Apply a threshold on the saturation
    cv::Mat mask;
    cv::inRange(I, cv::Scalar(0, 40, 0), cv::Scalar(180, 255, 255), mask);

    // Convert to grayscale and blur
    cv::Mat gray;
    if (mask.channels() == 3) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = mask.clone();
    }
    // define kernel size a 9 * downsampling factor and cast to the nearest odd value
    int kernel_size = static_cast<int>(9 * downsampling_factor);
    if (kernel_size % 2 == 0) {
        kernel_size += 1;
    }
    cv::GaussianBlur(gray, gray, cv::Size(kernel_size, kernel_size), downsampling_factor*2, downsampling_factor*2);

     // Find circles using Hough Transform
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                    1,     // dp
                    110*downsampling_factor,   // minDist
                    100, 30, // param1, param2
                    95*downsampling_factor, 210*downsampling_factor); // minRadius, maxRadius
    
    // Sort circles by radius in ascending order
    std::sort(circles.begin(), circles.end(), [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });

    return circles;
}
*/

std::vector<cv::Vec3f> get_circles_positions(const cv::Mat& I, const float downsampling_factor) {
    std::vector<cv::Vec3f> circles;
    cv::Mat mask;

    cv::Mat imgLab;
    cv::cvtColor(I, imgLab, cv::COLOR_BGR2Lab);

    // Split nei canali
    cv::Mat mask_lab;
    std::vector<cv::Mat> lab_planes(3);
    cv::split(imgLab, lab_planes);

    cv::Mat L = lab_planes[0];  // Lightness
    cv::Mat a = lab_planes[1];  // Green ↔ Red
    cv::Mat b = lab_planes[2];  // Blue ↔ Yellow

    //set a threshold in Lab format
    cv::inRange(a, 150, 255, a);
    cv::inRange(b, 138, 255, b);
    cv::bitwise_or(a, b, mask_lab); 

    // Morphological operations
    // Kernel definition for erosion
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, // shape
        cv::Size(7, 7)     // size
    );

    // Erosion (shrinks the white areas)
    cv::Mat eroded;
    cv::erode(mask_lab, eroded, kernel);

    // Dilation (expands the white areas)
    cv::Mat mask_lab_dilated;
    cv::dilate(eroded, mask_lab_dilated, cv::Mat(), cv::Point(-1,-1), 50); 
    
    // Threshold in HSV space
    cv::Mat mask_hsv;
    cv::cvtColor(I, mask_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(I, cv::Scalar(0, 40, 0), cv::Scalar(180, 255, 255), mask_hsv);

    // Refine HSV mask with Lab mask
    cv::Mat mask_hsv_refined;
    cv::bitwise_and(mask_hsv, mask_lab_dilated, mask_hsv_refined);

    // Combine both masks
    cv::bitwise_or(mask_lab, mask_hsv_refined, mask);

    // // uncomment to see the final mask *****
    // cv::namedWindow("Final Mask", cv::WINDOW_KEEPRATIO);
    // cv::imshow("Final Mask", mask);

    // Convert to grayscale and blur
    cv::Mat gray;
    if (mask.channels() == 3) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = mask.clone();
    }
    // define kernel size a 9 * downsampling factor and cast to the nearest odd value
    int kernel_size = static_cast<int>(9 * downsampling_factor);
    if (kernel_size % 2 == 0) {
        kernel_size += 1;
    }
    cv::GaussianBlur(gray, gray, cv::Size(kernel_size, kernel_size), downsampling_factor*2, downsampling_factor*2);

     // Find circles using Hough Transform
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT,
                    1,     // dp
                    200*downsampling_factor,   // minDist
                    100, 30, // param1, param2
                    95*downsampling_factor, 210*downsampling_factor); // minRadius, maxRadius
    
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

std::vector<DetectedCoin> get_positions_and_values_above_threshold(const cv::Mat& result, double threshold, double template_size, std::string label) {
    std::vector<DetectedCoin> positions_and_values;
    double radius = template_size / 2.0;

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            
            float value = result.at<float>(y, x);
            if (value >= threshold) {
                DetectedCoin new_coin;
                new_coin.center = cv::Point(x + radius, y + radius);
                new_coin.radius = radius;
                new_coin.confidence = value;
                new_coin.class_name = label;

                add_near_point(new_coin, positions_and_values, radius);
            }
        }
    }
    return positions_and_values;
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

bool add_near_point(const DetectedCoin& new_point, std::vector<DetectedCoin>& points, double min_distance) {
    cv::Point target = new_point.center;
    float radius = new_point.radius;
    float confidence = new_point.confidence;
    std::string label = new_point.class_name;

    for (auto& d : points) {
        cv::Point& p = d.center;
        float& r = d.radius;
        float& val = d.confidence;
        std::string& lbl = d.class_name;

        double dx = p.x - target.x;
        double dy = p.y - target.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        if (distance < min_distance){
            if (confidence >= val) {
                p = target;
                r = radius;
                val = confidence;
                lbl = label;
            }
            return true;
        }
    }
    points.push_back(new_point);
    return false;
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