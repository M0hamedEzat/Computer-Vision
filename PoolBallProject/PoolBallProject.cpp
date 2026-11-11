#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <cmath>

static cv::Mat tableHomography() {
    std::vector<cv::Point2f> src = { { 337.f, 35.f },{ 1888.f, 451.f },{ 1570.f, 21.f },{ 38.f, 472.f } };
    std::vector<cv::Point2f> dst = { { 0.f, 0.f },{ 1434.f, 683.f },{ 1434.f, 0.f },{ 0.f, 683.f } };
    return cv::getPerspectiveTransform(src, dst);
}

cv::Mat warpTable(const cv::Mat& inputImage) {
    cv::Mat warped;
    cv::warpPerspective(inputImage, warped, tableHomography(), cv::Size(1434, 683));
    return warped;
}


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

static std::vector<cv::Vec4f> findCirclesMultiChannel(const cv::Mat& roiBGR) {
    cv::Mat gray, hsv;
    cv::cvtColor(roiBGR, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(roiBGR, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> bgr, hsvCh;
    cv::split(roiBGR, bgr);
    cv::split(hsv, hsvCh);
    std::vector<cv::Mat> channels = { gray, bgr[0], bgr[1], bgr[2], hsvCh[1], hsvCh[2] };

    std::vector<cv::Vec3f> all;
    for (const cv::Mat& ch : channels) {
        cv::Mat blurred;
        cv::medianBlur(ch, blurred, 5);
        std::vector<cv::Vec3f> found;
        cv::HoughCircles(blurred, found, cv::HOUGH_GRADIENT, 1, 30, 80, 12, 19, 29);
        all.insert(all.end(), found.begin(), found.end());
    }

    // circles whose centres agree (< 22 px) are one ball; votes = channels that saw it
    std::vector<cv::Vec4f> merged;
    std::vector<bool> used(all.size(), false);
    for (size_t i = 0; i < all.size(); i++) {
        if (used[i]) continue;
        cv::Vec3f sum(0, 0, 0);
        int votes = 0;
        for (size_t j = i; j < all.size(); j++) {
            if (used[j]) continue;
            if (std::hypot(all[i][0] - all[j][0], all[i][1] - all[j][1]) < 22.0) {
                sum += all[j];
                used[j] = true;
                votes++;
            }
        }
        merged.emplace_back(sum[0] / votes, sum[1] / votes, sum[2] / votes, (float)votes);
    }
    return merged;
}

// ---------- split one cluster blob into individual balls ----------
static std::vector<cv::Point2f> splitCluster(const cv::Mat& original, const cv::Mat& H,
    const cv::Mat& clusterMask,  // table coords, this blob only
    const cv::Mat& foreground,   // table coords, all blobs, dilated
    double ballArea){
    // 1. region of the original image that contains this cluster
    cv::Rect r = cv::boundingRect(clusterMask);
    std::vector<cv::Point2f> corners = {
        { (float)r.x, (float)r.y },{ (float)r.br().x, (float)r.y },
        { (float)r.br().x, (float)r.br().y },{ (float)r.x, (float)r.br().y } };
    std::vector<cv::Point2f> origCorners;
    cv::perspectiveTransform(corners, origCorners, H.inv());
    cv::Rect b = cv::boundingRect(origCorners);
    cv::Rect roi(b.x - 30, b.y - 30, b.width + 60, b.height + 60);
    roi &= cv::Rect(0, 0, original.cols, original.rows);
    if (roi.area() == 0) return {};

    // 2. circles in the original image
    std::vector<cv::Vec4f> circles = findCirclesMultiChannel(original(roi));
    if (circles.empty()) return {};

    // 3. map centres to table coordinates; keep those on foreground seen by >= 2 channels
    std::vector<cv::Point2f> pts, tablePts;
    for (const auto& c : circles) pts.emplace_back(c[0] + roi.x, c[1] + roi.y);
    cv::perspectiveTransform(pts, tablePts, H);

    std::vector<cv::Point2f> centres;
    cv::Mat circleMask = cv::Mat::zeros(original.size(), CV_8UC1);
    for (size_t i = 0; i < circles.size(); i++) {
        int x = cvRound(tablePts[i].x), y = cvRound(tablePts[i].y);
        if (x < 0 || y < 0 || x >= foreground.cols || y >= foreground.rows) continue;
        if (circles[i][3] < 2 || foreground.at<uchar>(y, x) == 0) continue;
        centres.push_back(tablePts[i]);
        cv::circle(circleMask, cv::Point(cvRound(pts[i].x), cvRound(pts[i].y)),
            cvRound(0.9f * circles[i][2]), cv::Scalar(255), cv::FILLED);
    }
    if (centres.empty()) return {};

    // 4. whatever part of the cluster the circles do not explain is probably a missed ball
    cv::Mat explained, leftover;
    cv::warpPerspective(circleMask, explained, H, clusterMask.size(), cv::INTER_NEAREST);
    cv::bitwise_not(explained, explained);
    cv::bitwise_and(clusterMask, explained, leftover);
    cv::morphologyEx(leftover, leftover, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21)));
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(leftover, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : cs) {
        if (cv::contourArea(c) < 0.35 * ballArea) continue;
        cv::Moments m = cv::moments(c);
        centres.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
    }
    return centres;
}

std::vector<cv::Point2f> extractBlobCenters(const cv::Mat& edges, const cv::Mat& original, const cv::Mat& H){
    const int W = edges.cols, Ht = edges.rows;
    const int BAND = 45;
    const double ballArea = 3200.0;

    // 1. cut cushion lines, re-close gaps
    cv::Mat work = removeCushionLines(edges);
    cv::morphologyEx(work, work, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

    // 2. blank pocket mouths
    const std::vector<cv::Point> pockets = {
        { 0, 0 },{ W / 2, 0 },{ W, 0 },{ 0, Ht },{ W / 2, Ht },{ W, Ht } };
    for (size_t i = 0; i < pockets.size(); i++)
        cv::circle(work, pockets[i], (i == 1 || i == 4) ? 60 : 90, cv::Scalar(0), cv::FILLED);

    // 3. solid blobs (fill inside the table, convex hull near the border)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(work, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filled = cv::Mat::zeros(work.size(), CV_8UC1);
    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        bool nearBorder = r.x < BAND || r.y < BAND || r.br().x > W - BAND || r.br().y > Ht - BAND;
        if (nearBorder) {
            std::vector<cv::Point> hull;
            cv::convexHull(c, hull);
            if (cv::contourArea(hull) < 300) continue;
            cv::fillConvexPoly(filled, hull, cv::Scalar(255));
        }
        else {
            cv::drawContours(filled, std::vector<std::vector<cv::Point>>{c}, -1,
                cv::Scalar(255), cv::FILLED);
        }
    }

    // foreground (before the opening shaves balls) is used to validate circle detections
    cv::Mat foreground;
    cv::dilate(filled, foreground, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

    // 4. open: removes thin stuff + table spots
    cv::morphologyEx(filled, filled, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
    cv::imshow("Debug: Cleaned Filled Blobs", filled);

    // 5. one centre per blob; clusters are split with circle detection
    std::vector<cv::Point2f> centres;
    cv::findContours(filled, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < 0.3 * ballArea) continue;
        int n = std::max(1, (int)std::lround(area / ballArea));

        if (n == 1) {
            cv::Moments m = cv::moments(c);
            centres.emplace_back((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
            continue;
        }

        cv::Mat blob = cv::Mat::zeros(filled.size(), CV_8UC1);
        cv::drawContours(blob, std::vector<std::vector<cv::Point>>{c}, -1,
            cv::Scalar(255), cv::FILLED);

        std::vector<cv::Point2f> split = splitCluster(original, H, blob, foreground, ballArea);
        if (!split.empty()) {
            centres.insert(centres.end(), split.begin(), split.end());
            continue;
        }

        // fallback: k-means on the blob pixels (your previous behaviour)
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

    // 6. drop duplicates (the same ball seen by two routes)
    std::vector<cv::Point2f> unique;
    for (const auto& p : centres) {
        bool dup = false;
        for (const auto& q : unique)
            if (cv::norm(p - q) < 25.0) { dup = true; break; }
        if (!dup) unique.push_back(p);
    }
    return unique;
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

cv::Mat detectBalls(const cv::Mat& flatTable, const cv::Mat& original) {
    // Step 1: Preprocess and extract edges
    cv::Mat edges = createEdgeMask(flatTable);
    cv::imshow("Debug: Canny Edges", edges);

    // Step 2: Clean blobs and extract centers (handling clusters via K-Means)
    std::vector<cv::Point2f> centres = extractBlobCenters(edges, original, tableHomography());

    // Step 3: Filter out pockets, sample colors, and draw final visual results
    cv::Mat finalOutput = renderResults(flatTable, centres);

    return finalOutput;
}


// Required entry point function
void MyApplication() {
    std::cout << "=== Pool Ball System Initialized ===" << std::endl;

    std::string filename = "Table4.jpeg";
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
    cv::Mat foundBalls = detectBalls(flatTable, img);
    cv::imshow("Detected Balls", foundBalls);

    std::cout << "Press any key in the image window to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    MyApplication();
    return 0;
}