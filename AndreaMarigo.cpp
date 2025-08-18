
#include "main.h"

cv::Mat display_hist(cv::Mat I, int bins, std::string name) {
    cv::Mat hist;
    if (I.channels() > 1) {
        return hist;
    }

    int histSize[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    int channels[] = { 0 };

    int width = bins*2;
    int height = 400;

    cv::calcHist(&I, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    cv::Mat histImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(hist, hist, 0, height, cv::NORM_MINMAX);
    for (int i = 1; i < bins; i++) {
        cv::line(histImage,
                 cv::Point(2*(i-1), height - cvRound(hist.at<float>(i - 1))),
                 cv::Point(2*i, height - cvRound(hist.at<float>(i))),
                 cv::Scalar(255, 255, 255));
    }

    cv::namedWindow(name);
    cv::imshow(name, histImage);

    return hist;
}