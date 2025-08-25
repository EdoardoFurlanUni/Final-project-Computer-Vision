
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

// Funzione per ottenere un'immagine illuminant-invariant
cv::Mat computeIlluminantInvariant(const cv::Mat& img) {
    // Controllo: deve essere 3 canali (BGR)
    if (img.empty() || img.channels() != 3) {
        throw std::runtime_error("L'immagine deve essere a 3 canali BGR!");
    }

    // Converti in float32 normalizzato [0,1]
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

    // Split nei canali
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);
    cv::Mat B = channels[0];
    cv::Mat G = channels[1];
    cv::Mat R = channels[2];

    const float eps = 1e-6f; // per stabilità numerica

    // Calcola log(R/G) e log(B/G)
    cv::Mat R_over_G, B_over_G;
    cv::divide(R, G + eps, R_over_G);
    cv::divide(B, G + eps, B_over_G);

    cv::log(R_over_G + eps, R_over_G);
    cv::log(B_over_G + eps, B_over_G);

    // Combina i due canali in una matrice a 2 canali
    cv::Mat zeros = cv::Mat::zeros(R_over_G.size(), CV_32F);

    std::vector<cv::Mat> invariant_channels = {R_over_G, B_over_G, zeros};
    cv::Mat invariant;
    cv::merge(invariant_channels, invariant);

    // Normalizza per visualizzazione (0-1)
    cv::normalize(invariant, invariant, 0, 1, cv::NORM_MINMAX);

    return invariant;
}

cv::Mat contrast_stretching(cv::Mat I, int threshold, int max_coin_value) {
    // mask contains pixel of value <= threshold
    cv::Mat mask;
    cv::threshold(I, mask, threshold, 255, cv::THRESH_BINARY_INV);
    if (mask.type() != CV_8U) {
        mask.convertTo(mask, CV_8U);
    }
    // Multiply pixel by L/threshold
    cv::Mat temp;
    I.convertTo(temp, CV_32F);
    temp = temp * max_coin_value/threshold;
    temp.convertTo(temp, CV_8U);

    // Apply the mask
    cv::Mat stretched;
    stretched = cv::Mat::ones(I.size(), I.type()) * 255;       // all pixel to 255
    temp.copyTo(stretched, mask);                              // overwrite pixel < threshold

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

std::vector<cv::Mat> preprocess_images(const std::vector<cv::Mat>& images, float T, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image = img;

        // new_image = correct_illumination(new_image);
        new_image = contrast_stretching(img, T);

        cv::GaussianBlur(new_image, new_image, cv::Size(s, s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}
std::vector<cv::Mat> preprocess_images_test(const std::vector<cv::Mat>& images, float T, int s, float sigma) {
    std::vector<cv::Mat> processed_images;
    processed_images.reserve(images.size());

    for (const cv::Mat& img : images) {
        cv::Mat new_image = img;

        new_image = contrast_stretching(new_image, T);
        // new_image = correct_illumination(new_image);
        
        cv::GaussianBlur(new_image, new_image, cv::Size(s, s), sigma);

        processed_images.push_back(new_image);
    }

    return processed_images;
}