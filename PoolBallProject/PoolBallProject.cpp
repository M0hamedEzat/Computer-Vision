#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>

cv::Mat warpTable(const cv::Mat& inputImage) {
    // 1. Define the 4 corners in the original image (Column, Row)
    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(337.0f, 35.0f),    // Top-Left
        cv::Point2f(1888.0f, 451.0f),  // Bottom-Right
        cv::Point2f(1570.0f, 21.0f),   // Top-Right
        cv::Point2f(38.0f, 472.0f)     // Bottom-Left
    };

    // 2. Define the 4 target corners for the flattened output (Width = 1434, Height = 683)
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0.0f, 0.0f),           // Top-Left
        cv::Point2f(1434.0f, 683.0f),      // Bottom-Right
        cv::Point2f(1434.0f, 0.0f),        // Top-Right
        cv::Point2f(0.0f, 683.0f)          // Bottom-Left
    };

    // 3. Get the perspective transformation matrix
    cv::Mat matrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    // 4. Apply the warp to generate the final 1434x683 flat table image
    cv::Mat warpedImage;
    cv::warpPerspective(inputImage, warpedImage, matrix, cv::Size(1434, 683));

    return warpedImage;
}

cv::Mat isolateBalls(const cv::Mat& flatImage) {
    // 1. Convert the image from BGR (OpenCV's default) to HSV
    cv::Mat hsvImage;
    cv::cvtColor(flatImage, hsvImage, cv::COLOR_BGR2HSV);

    // 2. Define the HSV range for the green table cloth
    // These values might need slight tweaking, but this is a standard green range
    cv::Scalar lowerGreen(35, 50, 50);
    cv::Scalar upperGreen(85, 255, 255);

    // 3. Create a mask: Pixels in the green range become white (255), everything else black (0)
    cv::Mat greenMask;
    cv::inRange(hsvImage, lowerGreen, upperGreen, greenMask);

    // 4. We actually want the BALLS (not the table), so we invert the mask
    cv::Mat ballMask;
    cv::bitwise_not(greenMask, ballMask);

    return ballMask;
}


// Required entry point function
void MyApplication() {
    std::cout << "=== Pool Ball System Initialized ===" << std::endl;

    std::string filename = "Table1.jpeg";
    cv::Mat img = cv::imread(filename);

    if (img.empty()) {
        std::cerr << "Error: Could not load " << filename << "!" << std::endl;
        std::cerr << "Make sure your images are in the main project directory." << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << filename << std::endl;

    // <-- FIXED: Save the returned image into a new variable
    cv::Mat flatTable = warpTable(img);

    // <-- FIXED: Display the new 'flatTable' image instead of the original 'img'
    // cv::imshow("Flat Top-Down Table", flatTable);

    // show the balls without the table
    cv::Mat ballsOnly = isolateBalls(flatTable);
    cv::imshow("Isolated Balls (Mask)", ballsOnly);


    std::cout << "Press any key in the image window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    MyApplication();
    return 0;
}