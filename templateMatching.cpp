#include "main.h"

bool exists_near_point(const cv::Point& target, const std::vector<cv::Point>& points, const double min_distance) {

    for (const cv::Point& p : points) {

        double dx = p.x - target.x;
        double dy = p.y - target.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        if (distance < min_distance) {

            return true;
        }
    }
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