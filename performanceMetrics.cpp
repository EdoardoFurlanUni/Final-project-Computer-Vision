#include "main.h"

std::vector<std::vector<DetectedCoin>> get_labels_from_folder(const std::string& folder_path, const float downsampling_factor) {
    std::vector<std::vector<DetectedCoin>> all_labels;

    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {

        if (entry.is_regular_file() && entry.path().extension() == ".txt") {

            std::vector<DetectedCoin> labels;

            // read line by line
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {

                std::istringstream iss(line);
                DetectedCoin coin;

                char c1, c2, c3; // for brackets and comma
                if (iss >> coin.class_name       // EUR_xxx
                >> c1 >> coin.center.x           // '(' // x
                >> c2 >> coin.center.y           // ',' // y
                >> c3 >> coin.radius) {          // ')' // r

                    // Ignore line containing the sum
                    if (coin.class_name == "SUM") {
                        continue;
                    }

                    coin.center.x = static_cast<int>(coin.center.x * downsampling_factor);
                    coin.center.y = static_cast<int>(coin.center.y * downsampling_factor);
                    coin.radius = static_cast<int>(coin.radius * downsampling_factor);
                    coin.confidence = 1.0; // ground truth confidence
                    
                    labels.push_back(coin);
                }
            }

            all_labels.push_back(labels);
        }
    }
    return all_labels;
}

cv::Point2f intersection_and_union(const DetectedCoin label, const DetectedCoin prediction) {
    // Compute the intersection and union between the ground truth label and the predicted coin
    float intersection_area = 0.0;
    float union_area = 0.0;

    float dist_centers = cv::norm(label.center - prediction.center);

    // verify that the two circles intersects
    if (dist_centers > (label.radius + prediction.radius)) {
        return cv::Point2f(0.0f, 0.0f);
    }

    // Compute the intersection area
    if (dist_centers < std::abs(label.radius - prediction.radius)) {    // verify that the two circles are not completely overlapping

        intersection_area = CV_PI * std::pow(std::min(label.radius, prediction.radius), 2);
    } else {

        float alpha = 2.0f * acos((label.radius*label.radius + dist_centers*dist_centers - prediction.radius*prediction.radius) / (2.0f * label.radius * dist_centers));
        float beta  = 2.0f * acos((prediction.radius*prediction.radius + dist_centers*dist_centers - label.radius*label.radius) / (2.0f * prediction.radius * dist_centers));

        float area1 = 0.5f * label.radius*label.radius * (alpha - sin(alpha));
        float area2 = 0.5f * prediction.radius*prediction.radius * (beta  - sin(beta));
        intersection_area = area1 + area2;
    }

    // Compute the union area U = A1 + A2 - A_intersection
    union_area = CV_PI * (label.radius*label.radius + prediction.radius*prediction.radius) - intersection_area;

    // Compute the IoU
    return (union_area > 0) ? cv::Point2f(intersection_area, union_area) : cv::Point2f(0.0f, 0.0f);
}

cv::Point2f compute_mIoU_and_accuracy(const std::vector<DetectedCoin> ground_truth_labels, const std::vector<DetectedCoin> predicted_labels) {
    float total_intersection = 0.0f;
    float total_union = 0.0f;
    float accurate_predictions = 0.0f;

    // search indices contains the indices of the predicted labels to search for an intersection
    std::vector<int> search_indices;
    for (size_t i = 0; i < predicted_labels.size(); ++i) {
        search_indices.push_back(i);
    }

    for (const DetectedCoin& gt_coin : ground_truth_labels) {
        bool found_match = false;

        for (const int& i : search_indices) {
            const DetectedCoin& pred_coin = predicted_labels[i];
            auto [intersection_area, union_area] = intersection_and_union(gt_coin, pred_coin);

            if (intersection_area > 0) { // if there is intersection between label and prediction than it can't happen again
                total_intersection += intersection_area;
                total_union += union_area;

                // if this specific IoU is greater than 0.5 then count it as an accurate prediction
                if (intersection_area / union_area >= 0.5) {
                    accurate_predictions += 1;
                }

                // remove index i from the ones to search for
                search_indices.erase(std::remove(search_indices.begin(), search_indices.end(), i), search_indices.end());

                // If we found a match, we can stop searching
                found_match = true;
                break;
            }
        }

        // if no match was found add the ground truth area to the union
        if (!found_match) {
            total_union += CV_PI * gt_coin.radius * gt_coin.radius;
        }
    }

    // add all the predicted areas not matched to the union
    for (const int& i : search_indices) {
        const DetectedCoin& pred_coin = predicted_labels[i];
        total_union += CV_PI * pred_coin.radius * pred_coin.radius;
    }

    return cv::Point2f(total_intersection / total_union, accurate_predictions / ground_truth_labels.size());
}

float sum_coins(const std::vector<DetectedCoin>& coins) {
    float total = 0.0f;
    for (const DetectedCoin& coin : coins) {
        // extract last 3 char from class_name and cast to float
        float value = std::stof(coin.class_name.substr(4)) / 100.0f;
        total += value;
    }
    return total;
}