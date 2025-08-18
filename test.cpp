#include "main.h"

int main(int argc, const char* argv[])
{
    const std::string filename = "./dataset/images/1_CENT/IMG_18.jpg";
    const cv::Mat image = cv::imread(filename);
    
    cv::namedWindow("test image", cv::WINDOW_AUTOSIZE);
    cv::imshow("test image", image);
    cv::waitKey(0);

    std::cout << "si" << std::endl;
    
    return 0;
}

