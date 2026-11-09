#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>


cv::Mat detectBalls(const cv::Mat& flatTable) {
    cv::Mat gray, blurred;

    // 1. Convert to grayscale (HoughCircles requires a single-channel image)
    cv::cvtColor(flatTable, gray, cv::COLOR_BGR2GRAY);

    // 2. Add a slight blur to reduce background noise from the felt texture
    cv::medianBlur(gray, blurred, 5);

    // 3. Detect the circles
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
            1,      // dp
            38,     // minDist: CRITICAL FIX. Strictly prevents overlapping circles on the same ball.
            100,    // param1: Restored to standard edge sensitivity.
            13,     // param2: Low enough to catch the stretched cue ball, but clean.
            18,     // minRadius
            28);    // maxRadius (tightened back down)


    // 4. Draw the detected circles on a copy of the table for visualization
    cv::Mat display = flatTable.clone();
    for (size_t i = 0; i < circles.size(); i++) {
        // Extract center coordinates and radius
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // Draw a green dot in the center
        cv::circle(display, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        // Draw a red outline around the ball
        cv::circle(display, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    }

    std::cout << "Found " << circles.size() << " balls!" << std::endl;
    return display;
}



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


// Required entry point function
void MyApplication() {
    std::cout << "=== Pool Ball System Initialized ===" << std::endl;

    std::string filename = "Table3.jpeg";
    cv::Mat img = cv::imread(filename);

    if (img.empty()) {
        std::cerr << "Error: Could not load " << filename << "!" << std::endl;
        std::cerr << "Make sure your images are in the main project directory." << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << filename << std::endl;

    // Save the returned image into a new variable and show it
    cv::Mat flatTable = warpTable(img);
    // cv::imshow("Flat Top-Down Table", flatTable);

    // Find balls using shapes
    cv::Mat foundBalls = detectBalls(flatTable);
    cv::imshow("Detected Balls", foundBalls);

    std::cout << "Press any key in the image window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    MyApplication();
    return 0;
}