#include "main.h"

std::vector<std::tuple<cv::Point, float>> get_positions_and_values_above_threshold(const cv::Mat& result, double threshold) {
    std::vector<std::tuple<cv::Point, float>> positions_and_values;

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            
            float value = result.at<float>(y, x);
            if (value >= threshold) {
                positions_and_values.emplace_back(cv::Point(x, y), value);
            }
        }
    }
    return positions_and_values;
}

bool add_near_point(const std::tuple<cv::Point, float, float, std::string>& new_point, std::vector<std::tuple<cv::Point, float, float, std::string>>& points, double min_distance) {
    cv::Point target = std::get<0>(new_point);
    float radius = std::get<1>(new_point);
    float confidence = std::get<2>(new_point);
    std::string label = std::get<3>(new_point);

    for (auto& d : points) {
        cv::Point& p = std::get<0>(d);
        float& r = std::get<1>(d);
        float& val = std::get<2>(d);
        std::string& lbl = std::get<3>(d);

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
    points.push_back(std::make_tuple(target, radius, confidence, label));
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