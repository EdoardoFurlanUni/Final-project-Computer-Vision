#include "main.h"

std::vector<cv::Mat> images;
int slider = 2;
int maxSlider = 100;

void on_trackbar(int pos, void* userdata) {
    std::vector<cv::Mat> concatenated_images;
    int ref_width = 700;
    int ref_height = 700;

    // aggiorna tutti i parametri delle trackbar
    slider = cv::getTrackbarPos("slider", "process");
    // tileGridSize = cv::getTrackbarPos("Tile Grid Size", "process");

    for (cv::Mat image : images) {
        cv::Mat processed_image = image.clone();
        
        // processa
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<cv::Point2f> points_contrast_stretching = {cv::Point2f(0,0), cv::Point2f(0.9*255, 255), cv::Point2f(255, 255)};
        processed_image = contrast_stretching(processed_image, points_contrast_stretching);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3, cv::Size(15, 15));
        clahe->apply(processed_image, processed_image);
        cv::GaussianBlur(processed_image, processed_image, cv::Size(3,3), 1.0);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        // resize images for concatenation
        cv::resize(image, image, cv::Size(ref_width, ref_height));
        cv::resize(processed_image, processed_image, cv::Size(ref_width, ref_height));

        // concatenate horizontally
        cv::Mat concatenatedH;
        cv::hconcat(image, processed_image, concatenatedH);
        concatenated_images.push_back(concatenatedH);
    }

    // concatenate vertically
    cv::Mat concatenatedV;
    cv::vconcat(concatenated_images, concatenatedV);
    cv::imshow("process", concatenatedV);
}

int main(int argc, const char* argv[]) {

    std::vector<std::string> filenames = {"../template/images/10_CENT/IMG_22_temp.jpg", "../test/images/IMG_24.jpg"};
    cv::namedWindow("process", cv::WINDOW_KEEPRATIO);

    // Load images
    for (const auto& filename : filenames) {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    // Cut images
    std::vector<cv::Rect> cuts = get_bbox_containing_coins(images, 50);
    images = cut_images(images, cuts);

    // downsample images
    const float downsampling_factor = 0.5;
    for (cv::Mat& image : images) {
        cv::resize(image, image, cv::Size(), downsampling_factor, downsampling_factor);
    }

    // Create trackbars
    cv::createTrackbar("slider", "process", NULL, maxSlider, on_trackbar);

    // Initial call to display image
    on_trackbar(0, 0);

    cv::waitKey(0);

    // const std::vector<std::string> dataset_images_paths = {
    //     "../dataset/images/1_CENT", "../dataset/images/2_CENT", "../dataset/images/10_CENT", "../dataset/images/20_CENT", "../dataset/images/50_CENT", "../dataset/images/1_EURO", "../dataset/images/2_EURO"
    // };

    // for (int i = 0; i < dataset_images_paths.size(); i++) {
    //     std::string folder = dataset_images_paths[i];
    //     std::vector<std::string> file_paths = get_file_names(folder);

    //     for (const std::string& file : file_paths) {
    //         cv::Mat img = cv::imread(file);
    //         if (!img.empty()) {
    //             // display size
    //             std::cout << "Image " << file << ": " << img.size() << std::endl;

    //             // se la dimensione non è 3024 x 4032, ridimensiona
    //             if (img.size() != cv::Size(3024, 4032)) {
    //                 cv::resize(img, img, cv::Size(3024, 4032));
    //                 std::cout << "Resized image size: " << img.size() << std::endl;
    //                 // salva l'immagine
    //                 cv::imwrite(file, img);
    //             }

    //         } else {
    //             std::cerr << "Impossible to read: " << file << std::endl;
    //         }
    //     }
    // }

    // const std::string test_images_path = "../test/images/";

    // std::vector<cv::Mat> test_images = load_images_from_folder(test_images_path);

    // for (const auto& image : test_images) {
    //     // display size
    //     std::cout << "Image size: " << image.size() << std::endl;
    // }

    return 0;
}
