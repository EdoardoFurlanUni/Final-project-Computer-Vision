#include "main.h"

std::vector<cv::Mat> load_images_from_folder(const std::string& folder, int flags) {
    std::vector<cv::Mat> images_in_folder;
   
    DIR* dir = opendir(folder.c_str());

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;

        if (filename == "." || filename == "..") continue;

        std::string full_path = folder + "/" + filename;
        cv::Mat img = cv::imread(full_path, flags);
        if (!img.empty()) {
            images_in_folder.push_back(img);
        } else {
            std::cerr << "Error loading: " << full_path << std::endl;
        }
    }
    closedir(dir);

    return images_in_folder;
}
