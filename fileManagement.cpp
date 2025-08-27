#include "main.h"

std::vector<std::string> get_file_names(const std::string& folder) {
    std::vector<std::string> filenames;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                filenames.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    return filenames;
}

std::vector<cv::Mat> load_images_from_folder(const std::string& folder, int flags) {
    std::vector<cv::Mat> images_in_folder;

    std::vector<std::string> file_paths = get_file_names(folder);

    for (const std::string& path : file_paths) {
        cv::Mat img = cv::imread(path, flags);
        if (!img.empty()) {
            images_in_folder.push_back(img);
        } else {
            std::cerr << "Impossible to read: " << path << std::endl;
        }
    }

    return images_in_folder;
}
