
#include "main.h"

//MODIFIED

cv::Mat correct_illumination(cv::Mat I){
    cv::Mat illumination, corrected;

    // apply a Gaussian filter with a large kernel to estimate the illumination
    cv::GaussianBlur(I, illumination, cv::Size(101, 101), 0);

    // convert the images to float for division
    I.convertTo(I, CV_32F);
    illumination.convertTo(illumination, CV_32F);
    cv::Scalar meanVal = cv::mean(illumination);
    
    // divide the original image by the illumination
    // and multiply by the mean value of the illumination (normalization)
    cv::divide(I, illumination, corrected);
    corrected *= cv::mean(illumination)[0];
    // std::cout << "Mean value before conversion: " << cv::mean(illumination)[0] << std::endl;
    corrected.convertTo(corrected, CV_8U);

    return corrected;
}

cv::Mat contrast_stretching(const cv::Mat& I, const std::vector<cv::Point2f>& points) {
    cv::Mat stretched = cv::Mat::zeros(I.size(), I.type());

    for (int i = 1; i < points.size(); ++i) {
        cv::Point2f A = points[i-1];
        cv::Point2f B = points[i];

        // Crea maschera per pixel nell'intervallo [A.x, B.x]
        cv::Mat mask;
        cv::inRange(I, cv::Scalar(A.x), cv::Scalar(B.x), mask);

        // Applica la trasformazione lineare solo ai pixel selezionati
        for (int r = 0; r < I.rows; ++r) {
            for (int c = 0; c < I.cols; ++c) {
                if (mask.at<uchar>(r, c)) {
                    float val = I.at<uchar>(r, c);
                    float new_val = A.y + (val - A.x) * (B.y - A.y) / (B.x - A.x);
                    stretched.at<uchar>(r, c) = cv::saturate_cast<uchar>(new_val);
                }
            }
        }
    }

    return stretched;
}

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

std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, const std::vector<cv::Point2f>& points, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image = img;

        // new_image = correct_illumination(new_image);
        new_image = contrast_stretching(img, points);

        cv::GaussianBlur(new_image, new_image, cv::Size(s, s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}
std::vector<cv::Mat> preprocess_images_test(const std::vector<cv::Mat>& images, const std::vector<cv::Point2f>& points, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image = img;

        new_image = contrast_stretching(new_image, points);
        // new_image = correct_illumination(new_image);
        
        cv::GaussianBlur(new_image, new_image, cv::Size(s, s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}