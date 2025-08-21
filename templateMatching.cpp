#include "main.h"

bool exists_near_point(const cv::Point& target, const std::vector<cv::Point>& points, const double min_distance) {

    for (const cv::Point& p : points) {

        double dx = p.x - target.x;
        double dy = p.y - target.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        if (distance < min_distance) {

            return true; // trovato un punto vicino
        }
    }
    return false; // nessun punto vicino
}