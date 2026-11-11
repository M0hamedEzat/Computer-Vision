#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>


cv::Mat createEdgeMask(const cv::Mat& flatTable) {
    cv::Mat gray, blurred, edges;
    cv::cvtColor(flatTable, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, blurred, 5);

    cv::Canny(blurred, edges, 30, 100);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6)));

    // Close small gaps in the rings
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

    return edges;
}

// Removes long straight strokes (cushion lines) so they can't glue onto balls.
// L must be longer than a ball's height (~85 px) but shorter than a cushion segment.
static cv::Mat removeCushionLines(const cv::Mat& mask, int L = 120) {
    cv::Mat horiz, vert, lines;
    cv::morphologyEx(mask, horiz, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(L, 1)));
    cv::morphologyEx(mask, vert, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, L)));
    cv::bitwise_or(horiz, vert, lines);
    cv::dilate(lines, lines, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    cv::Mat out = mask.clone();
    out.setTo(0, lines);
    return out;
}

std::vector<cv::Point2f> extractBlobCenters(const cv::Mat& edges) {
    const int W = edges.cols, H = edges.rows;
    const int BAND = 45;   // components this close to the border get the hull treatment

    // 1. cut the cushion lines out, then re-close the small gaps they leave
    cv::Mat work = removeCushionLines(edges);
    cv::morphologyEx(work, work, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

    // 2. blank the pocket mouths (fixed table geometry, stationary camera)
    const std::vector<cv::Point> pockets = {
        { 0, 0 },{ W / 2, 0 },{ W, 0 },{ 0, H },{ W / 2, H },{ W, H } };
    for (size_t i = 0; i < pockets.size(); i++) {
        bool middle = (i == 1 || i == 4);
        cv::circle(work, pockets[i], middle ? 60 : 90, cv::Scalar(0), cv::FILLED);
    }

    // 3. solid blobs: normal fill inside the table, convex hull near the border
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filled = cv::Mat::zeros(work.size(), CV_8UC1);
    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        bool nearBorder = r.x < BAND || r.y < BAND ||
            r.br().x > W - BAND || r.br().y > H - BAND;
        if (nearBorder) {
            std::vector<cv::Point> hull;
            cv::convexHull(c, hull);
            if (cv::contourArea(hull) < 300) continue;      // specks
            cv::fillConvexPoly(filled, hull, cv::Scalar(255));
        }
        else {
            cv::drawContours(filled, std::vector<std::vector<cv::Point>>{c}, -1,
                cv::Scalar(255), cv::FILLED);
        }
    }

    // 4. open: removes leftover thin stuff + table spots
    cv::morphologyEx(filled, filled, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
    cv::imshow("Debug: Cleaned Filled Blobs", filled);

    // 5. centres + k-means split (unchanged from your version)
    const double ballArea = 3200.0;
    std::vector<cv::Point2f> centres;

    cv::findContours(filled, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < 0.3 * ballArea) continue;
        int n = std::max(1, (int)std::lround(area / ballArea));

        if (n == 1) {
            cv::Moments m = cv::moments(c);
            centres.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
        }
        else {
            cv::Mat blob = cv::Mat::zeros(filled.size(), CV_8UC1);
            cv::drawContours(blob, std::vector<std::vector<cv::Point>>{c}, -1,
                cv::Scalar(255), cv::FILLED);
            std::vector<cv::Point> pts;
            cv::findNonZero(blob, pts);
            cv::Mat data((int)pts.size(), 2, CV_32F);
            for (int i = 0; i < (int)pts.size(); i++) {
                data.at<float>(i, 0) = (float)pts[i].x;
                data.at<float>(i, 1) = (float)pts[i].y;
            }
            cv::Mat labels, km;
            cv::kmeans(data, n, labels,
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.5),
                5, cv::KMEANS_PP_CENTERS, km);
            for (int i = 0; i < n; i++)
                centres.emplace_back(km.at<float>(i, 0), km.at<float>(i, 1));
        }
    }
    return centres;
}

cv::Mat renderResults(const cv::Mat& flatTable, const std::vector<cv::Point2f>& centres) {
    cv::Mat display = flatTable.clone();
    cv::Mat hsvTable;
    cv::cvtColor(flatTable, hsvTable, cv::COLOR_BGR2HSV);

    // 6 Pockets on a 1434x683 warped table
    std::vector<cv::Point> pockets = {
        cv::Point(0, 0), cv::Point(717, 0), cv::Point(1434, 0),
        cv::Point(0, 683), cv::Point(717, 683), cv::Point(1434, 683)
    };

    int validCount = 0;
    for (const auto& center : centres) {
        // Pocket Exclusion Filter
        bool isPocket = false;
        for (const auto& pocket : pockets) {
            if (cv::norm(center - cv::Point2f(pocket)) < 65.0) {
                isPocket = true;
                break;
            }
        }
        if (isPocket) continue;

        // Dynamic Color Sampling
        cv::Rect roi((int)center.x - 2, (int)center.y - 2, 5, 5);
        cv::Scalar ringColor(0, 0, 255); // Red fallback

        if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width < flatTable.cols && roi.y + roi.height < flatTable.rows) {
            cv::Mat colorPatch = hsvTable(roi);
            int val = (int)cv::mean(colorPatch)[2];

            if (val < 50) {
                ringColor = cv::Scalar(0, 0, 0); // Black for 8-ball
            }
            else {
                ringColor = cv::mean(flatTable(roi)); // Ball's BGR color
            }
        }

        // Draw center dot and colored ring
        cv::circle(display, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(display, center, 21, ringColor, 2, cv::LINE_AA);
        validCount++;
    }

    std::cout << "Successfully rendered " << validCount << " valid balls!" << std::endl;
    return display;
}

cv::Mat detectBalls(const cv::Mat& flatTable) {
    // Step 1: Preprocess and extract edges
    cv::Mat edges = createEdgeMask(flatTable);
    cv::imshow("Debug: Canny Edges", edges);

    // Step 2: Clean blobs and extract centers (handling clusters via K-Means)
    std::vector<cv::Point2f> centres = extractBlobCenters(edges);

    // Step 3: Filter out pockets, sample colors, and draw final visual results
    cv::Mat finalOutput = renderResults(flatTable, centres);

    return finalOutput;
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

    std::string filename = "Table1.jpeg";
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