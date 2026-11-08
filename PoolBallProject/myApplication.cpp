#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Required entry point function
void MyApplication() {
    std::cout << "=== Pool Ball System Initialized ===" << std::endl;

    // Load one of the test images
    std::string filename = "Table1.jpeg";
    cv::Mat img = cv::imread(filename);

    if (img.empty()) {
        std::cerr << "Error: Could not load " << filename << "!" << std::endl;
        std::cerr << "Make sure your images are in the main project directory." << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << filename << " (" << img.cols << "x" << img.rows << ")" << std::endl;

    // Display image to test OpenCV UI
    cv::imshow("Test Window", img);
    std::cout << "Press any key in the image window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// Standard main function to call your MyApplication() function
int main() {
    MyApplication();
    return 0;
}