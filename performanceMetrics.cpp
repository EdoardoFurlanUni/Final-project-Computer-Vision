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

float intersection_over_union(const DetectedCoin label, const DetectedCoin prediction) {
    // Compute the intersection over union (IoU) between the ground truth label and the predicted coin
    float intersection_area = 0.0;
    float union_area = 0.0;

    float dist_centers = cv::norm(label.center - prediction.center);

    // verify that the two circles intersects
    if (dist_centers > (label.radius + prediction.radius)) {
        return 0.0;
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
    return (union_area > 0) ? (intersection_area / union_area) : 0;
}

float compute_mIoU(const std::vector<DetectedCoin> ground_truth_labels, const std::vector<DetectedCoin> predicted_labels) {
    float total_mIoU = 0.0;
    int n = ground_truth_labels.size();

    // Compute IoU for each pair of ground truth and predicted coins
    for (const auto& gt_coin : ground_truth_labels) {
        float best_iou = 0.0;

        for (const auto& pred_coin : predicted_labels) {
            float iou = intersection_over_union(gt_coin, pred_coin);
            best_iou = std::max(best_iou, iou);
        }

        total_mIoU += best_iou;
    }
    

    return total_mIoU / n;
}