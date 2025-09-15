cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)

# Trova OpenCV installato da Homebrew
find_package(OpenCV REQUIRED)

add_executable(main main.cpp fileManagement.cpp preprocessing.cpp templateMatching.cpp performanceMetrics.cpp)

# Linka le librerie OpenCV
target_link_libraries(main ${OpenCV_LIBS})
