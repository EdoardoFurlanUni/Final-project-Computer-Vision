cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)

# Trova OpenCV installato da Homebrew
find_package(OpenCV REQUIRED)

add_executable(opencv_test main.cpp fileManagement.cpp preprocessing.cpp templateMatching.cpp svm.cpp)

# Linka le librerie OpenCV
target_link_libraries(opencv_test ${OpenCV_LIBS})

# Secondo eseguibile
add_executable(prova_preprocessing prova_preprocessing.cpp fileManagement.cpp preprocessing.cpp templateMatching.cpp)
target_link_libraries(prova_preprocessing ${OpenCV_LIBS})

target_include_directories(prova_preprocessing PRIVATE ${OpenCV_INCLUDE_DIRS})
