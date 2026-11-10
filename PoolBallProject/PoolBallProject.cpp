#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>


cv::Mat detectBalls(const cv::Mat& flatTable) {
    cv::Mat gray, blurred, edges;
    cv::cvtColor(flatTable, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, blurred, 5);

    // 1. CANNY EDGE MASK
    cv::Canny(blurred, edges, 50, 150);

    // 2. THICKEN THE MASK (Your suggestion to increase the circles)
    // We use a 5x5 brush to dilate the 1-pixel Canny lines into thick, forgiving bands.
    // This allows our perfect circle template to overlap heavily with oval-shaped balls.
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6));
    cv::dilate(edges, edges, dilateKernel);

    // 3. THINNER FEATURE TEMPLATE
    // Thinned the line thickness down to 1 pixel as you suggested.
    cv::Mat templateMask = cv::Mat::zeros(50, 50, CV_8UC1);
    cv::circle(templateMask, cv::Point(25, 25), 21, cv::Scalar(255), 5, cv::LINE_AA);

    // 4. FEATURE RECOGNITION
    cv::Mat matchResult;
    cv::matchTemplate(edges, templateMask, matchResult, cv::TM_CCOEFF_NORMED);

    // 5. LESS STRICT DETECTION
    std::vector<cv::Point> detections;
    // Lowered from 0.35 to 0.22. Now, even a broken, partial curve gets flagged as a ball.
    float matchThreshold = 0.35f;

    for (int y = 0; y < matchResult.rows; y++) {
        for (int x = 0; x < matchResult.cols; x++) {
            if (matchResult.at<float>(y, x) > matchThreshold) {
                detections.push_back(cv::Point(x + 25, y + 25));
            }
        }
    }

    // 6. NON-MAXIMUM SUPPRESSION
    // Prevent multiple detections on the exact same ball
    std::vector<cv::Point> finalBalls;
    for (const auto& pt : detections) {
        bool isDuplicate = false;
        for (const auto& finalPt : finalBalls) {
            if (cv::norm(pt - finalPt) < 25.0) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            finalBalls.push_back(pt);
        }
    }

    cv::Mat display = flatTable.clone();
    int validCount = 0;


    for (const auto& center : finalBalls) {
        bool isPocket = false;
        if (isPocket) continue;

        // Draw valid balls
        cv::circle(display, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(display, center, 21, cv::Scalar(0, 0, 255), 3, 8, 0);
        validCount++;
    }

    cv::imshow("Debug: Thickened Canny Mask", edges);
    cv::imshow("Debug: Thinner Feature Template", templateMask);
    std::cout << "Found " << validCount << " valid balls via Template Matching!" << std::endl;

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

    std::string filename = "Table2.jpeg";
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