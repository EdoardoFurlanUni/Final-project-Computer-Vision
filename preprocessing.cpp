
#include "main.h"

cv::Mat contrast_stretching(const cv::Mat& I, const std::vector<cv::Point2f>& points) {
    cv::Mat stretched;
    I.convertTo(stretched, CV_32F);

    cv::Mat previous_mask = cv::Mat::zeros(I.size(), CV_8U);

    for (int i = 1; i < points.size(); ++i) {
        int x1 = cvRound(points[i-1].x);
        int x2 = cvRound(points[i].x);
        int y1 = cvRound(points[i-1].y);
        int y2 = cvRound(points[i].y);

        // pixel <= x2
        cv::Mat mask;
        cv::threshold(I, mask, x2, 255, cv::THRESH_BINARY_INV);

        // isolate interval (x1, x2]
        cv::Mat regionMask;
        cv::bitwise_xor(mask, previous_mask, regionMask); 

        // transform pixels in this region
        cv::Mat I_region;
        I.copyTo(I_region, regionMask);
        I_region.convertTo(I_region, CV_32F);

        // apply linear transformation
        I_region = (I_region - x1) * ((y2 - y1) / static_cast<float>(x2 - x1)) + y1;
        I_region.convertTo(I_region, CV_8U);

        // write to destination
        I_region.copyTo(stretched, regionMask);

        // update previous_mask
        previous_mask = mask.clone();
    }

    return stretched;
}

std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, const std::vector<cv::Point2f>& points, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image;
        new_image = img.clone();

        // verify it is in gray scale
        if (img.channels() != 1) {
            std::cerr << "Image is not in grayscale format." << std::endl;
            return {};
        }

        new_image = contrast_stretching(new_image, points);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(15, 15));
        clahe->apply(new_image, new_image);
        cv::GaussianBlur(new_image, new_image, cv::Size(s,s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}

double getSSIM(const cv::Mat& img1, const cv::Mat& img2) {

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat numerator = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    cv::Mat denominator = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);
    cv::Scalar mssim = cv::mean(ssim_map);

    return mssim.val[0];
}
