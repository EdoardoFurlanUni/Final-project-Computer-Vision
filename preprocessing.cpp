
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
    cv::Mat stretched;
    I.convertTo(stretched, CV_32F);
    //  = cv::Mat::zeros(I.size(), I.type());

    cv::Mat previous_mask = cv::Mat::zeros(I.size(), CV_8U);

    for (int i = 1; i < points.size(); ++i) {
        int x1 = cvRound(points[i-1].x);
        int x2 = cvRound(points[i].x);
        int y1 = cvRound(points[i-1].y);
        int y2 = cvRound(points[i].y);

        // maschera: pixel <= x2
        cv::Mat mask;
        cv::threshold(I, mask, x2, 255, cv::THRESH_BINARY_INV);

        // isoliamo l’intervallo (x1, x2]
        cv::Mat regionMask;
        cv::bitwise_xor(mask, previous_mask, regionMask); 

        // trasformiamo i pixel di questa regione
        cv::Mat I_region;
        I.copyTo(I_region, regionMask);
        I_region.convertTo(I_region, CV_32F);

        // applica la trasformazione lineare
        I_region = (I_region - x1) * ((y2 - y1) / static_cast<float>(x2 - x1)) + y1;
        I_region.convertTo(I_region, CV_8U);

        // scrivi nella destinazione
        I_region.copyTo(stretched, regionMask);

        // aggiorna previous_mask
        previous_mask = mask.clone();
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
        cv::Mat new_image;
        new_image = img.clone();

        new_image = contrast_stretching(new_image, points);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(15, 15));
        clahe->apply(new_image, new_image);
        cv::GaussianBlur(new_image, new_image, cv::Size(3,3), 1.0);

        processed_images.push_back(new_image);
    }

    return processed_images;
}
std::vector<cv::Mat> preprocess_images_test(const std::vector<cv::Mat>& images, const std::vector<cv::Point2f>& points, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image;
        new_image = img.clone();

        new_image = contrast_stretching(new_image, points);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(15, 15));
        clahe->apply(new_image, new_image);
        cv::GaussianBlur(new_image, new_image, cv::Size(3,3), 1.0);

        processed_images.push_back(new_image);
    }

    return processed_images;
}