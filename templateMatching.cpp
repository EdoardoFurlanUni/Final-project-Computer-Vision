#include "main.h"

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

double estimateRotationAngle(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2, const std::vector<cv::DMatch>& matches) {
    std::vector<double> angles;
    for (auto& m : matches) {
        double angle = kp2[m.trainIdx].angle - kp1[m.queryIdx].angle;
        angles.push_back(angle);
    }
    if (angles.empty()) return 0.0;
    double sum = 0;
    for (auto& a : angles) sum += a;
    return sum / angles.size();
}

cv::Mat rotateImage(const cv::Mat& src, double angle) {
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return dst;
}

double computeSIFTConfidence(const std::vector<cv::KeyPoint>& kp_template, const std::vector<cv::KeyPoint>& kp_scene, const std::vector<cv::DMatch>& matches) {
    if (matches.empty()) {
        return 0.0;
    }

    // minimum distance
    double min_dist = 1e9;
    for (const auto& m : matches) {
        if (m.distance < min_dist) min_dist = m.distance;
    }

    // threshold to consider a match as "good"
    double threshold = std::max(2.0 * min_dist, 30.0);

    double total_distance = 0.0;
    int good_matches_count = 0;

    for (const auto& match : matches) {
        if (match.distance <= threshold) {
            total_distance += match.distance;
            good_matches_count++;
        }
    }

    if (good_matches_count == 0) return 0.0;

    double average_distance = total_distance / good_matches_count;
    double confidence = 1.0 / (1.0 + average_distance); // Normalize to [0, 1]

    return confidence;
}
