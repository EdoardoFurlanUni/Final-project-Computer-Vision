#include "main.h"

std::vector<cv::Mat> load_images_from_folder(const std::string& folder, int flags) {
    std::vector<cv::String> filenames;
    std::vector<cv::Mat> images;

    // Prende tutte le immagini (jpg, png, ecc.) nella cartella
    cv::glob(folder, filenames, false);

    // Carica le immagini nell'ordine restituito da glob (già ordinato lessicograficamente)
    for (const auto& file : filenames) {
        cv::Mat img = cv::imread(file, flags);
        if (!img.empty()) {
            images.push_back(img);
        } else {
            std::cerr << "⚠️  Impossibile caricare immagine: " << file << std::endl;
        }
    }
    return images;
}
