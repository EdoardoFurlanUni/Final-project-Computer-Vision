#include "main.h"

int main(int argc, const char* argv[]) {
    cv::Mat I = cv::imread("../test/videos/video1_frame/images/frame_452.jpg", cv::IMREAD_COLOR);

    std::vector<std::vector<DetectedCoin>> ground_truth_labels = get_labels_from_folder("../test/videos/video1_frame/labels", 1);

    std::vector<DetectedCoin> labels = ground_truth_labels[3];

    // Draw ground truth coins
    for (const auto& coin: labels) {
        cv::circle(I, coin.center, coin.radius, cv::Scalar(255, 255, 255), static_cast<int>(5*0.5), cv::LINE_AA);
        cv::putText(I, coin.class_name, cv::Point(coin.center.x-coin.radius/2, coin.center.y-coin.radius/2), cv::FONT_HERSHEY_SIMPLEX, 1.5*0.5, cv::Scalar(255, 255, 255), static_cast<int>(5*0.5));
    }

    cv::namedWindow("Results", cv::WINDOW_KEEPRATIO);
    cv::imshow("Results", I);
    cv::waitKey(0);

    return 0;
}
