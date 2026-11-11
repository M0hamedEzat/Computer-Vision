#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>

// --- AUTO-GENERATED GROUND TRUTH DATA ---
// Map of Image Filename -> Vector of exact (X, Y) pixel coordinates of the balls
std::map<std::string, std::vector<cv::Point2f>> allGroundTruths = {
    { "AllBalls.jpeg",{
        { 908.0, 253.0 },{ 959.0, 250.0 },{ 1013.0, 255.0 },{ 1072.0, 256.0 },{ 1132.0, 254.0 },{ 1184.0, 262.0 },{ 1244.0, 256.0 },{ 1307.0, 253.0 },{ 901.0, 329.0 },{ 961.0, 330.0 },{ 1013.0, 335.0 },{ 1067.0, 336.0 },{ 1130.0, 334.0 },{ 1187.0, 345.0 },{ 1245.0, 342.0 },{ 797.0, 293.0 }
    } },
    { "Table1.jpeg",{
        { 1092.0, 309.0 },{ 1212.0, 429.0 },{ 1131.0, 383.0 },{ 1050.0, 335.0 },{ 1211.0, 285.0 },{ 1170.0, 261.0 },{ 1173.0, 352.0 },{ 1132.0, 331.0 },{ 1210.0, 240.0 },{ 1170.0, 310.0 },{ 1169.0, 407.0 },{ 1135.0, 281.0 },{ 1212.0, 374.0 },{ 1091.0, 360.0 },{ 1213.0, 331.0 },{ 29.0, 288.0 }
    } },
    { "Table2.jpeg",{
        { 1103.0, 282.0 },{ 1224.0, 620.0 },{ 1121.0, 391.0 },{ 1046.0, 315.0 },{ 1305.0, 310.0 },{ 621.0, 202.0 },{ 1173.0, 361.0 },{ 1132.0, 331.0 },{ 725.0, 177.0 },{ 1269.0, 185.0 },{ 798.0, 665.0 },{ 438.0, 296.0 },{ 1235.0, 53.0 },{ 1090.0, 362.0 },{ 1205.0, 399.0 },{ 1169.0, 507.0 }
    } },
    { "Table3.jpeg",{
        { 908.0, 23.0 },{ 1224.0, 616.0 },{ 1199.0, 447.0 },{ 29.0, 336.0 },{ 1406.0, 305.0 },{ 621.0, 210.0 },{ 1294.0, 260.0 },{ 1252.0, 226.0 },{ 725.0, 177.0 },{ 1270.0, 171.0 },{ 798.0, 665.0 },{ 351.0, 249.0 },{ 1233.0, 53.0 },{ 901.0, 169.0 },{ 1201.0, 399.0 },{ 29.0, 434.0 }
    } },
    { "Table4.jpeg",{
        { 908.0, 23.0 },{ 1407.0, 546.0 },{ 1204.0, 444.0 },{ 1406.0, 305.0 },{ 621.0, 210.0 },{ 1294.0, 260.0 },{ 1252.0, 226.0 },{ 724.0, 181.0 },{ 1270.0, 171.0 },{ 798.0, 665.0 },{ 351.0, 249.0 },{ 1231.0, 58.0 },{ 901.0, 169.0 },{ 1201.0, 399.0 },{ 1392.0, 88.0 }
    } },
    { "Table5.jpeg",{
        { 23.0, 535.0 },{ 1407.0, 546.0 },{ 1204.0, 444.0 },{ 1406.0, 305.0 },{ 25.0, 131.0 },{ 1294.0, 260.0 },{ 1252.0, 226.0 },{ 72.0, 538.0 },{ 1270.0, 171.0 },{ 798.0, 665.0 },{ 351.0, 249.0 },{ 901.0, 169.0 },{ 1201.0, 399.0 },{ 26.0, 608.0 }
    } },
    { "Table6.jpeg",{
        { 23.0, 535.0 },{ 1364.0, 477.0 },{ 1204.0, 444.0 },{ 1406.0, 305.0 },{ 25.0, 131.0 },{ 1294.0, 260.0 },{ 1252.0, 226.0 },{ 72.0, 538.0 },{ 1270.0, 171.0 },{ 351.0, 249.0 },{ 901.0, 169.0 },{ 1201.0, 399.0 },{ 947.0, 629.0 }
    } }
};



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

std::vector<cv::Point2f> detectBalls(const cv::Mat& flatTable, const cv::Mat& original) {    // Step 1: Preprocess and extract edges
    cv::Mat edges = createEdgeMask(flatTable);
    cv::imshow("Debug: Canny Edges", edges);

    // Step 2: Clean blobs and extract centers (handling clusters via K-Means)
    std::vector<cv::Point2f> centres = extractBlobCenters(edges, original, tableHomography());

    // Step 3: Filter out pockets, sample colors, and draw final visual results
    cv::Mat finalOutput = renderResults(flatTable, centres);
    cv::imshow("Detected Balls", finalOutput);
    return centres;
}

struct EvaluationMetrics {
    int TP = 0, FP = 0, FN = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;
};

EvaluationMetrics evaluateDetections(const std::vector<cv::Point2f>& detected,
    const std::vector<cv::Point2f>& groundTruth,
    float threshold = 25.0f) { // 25px radius of leniency
    EvaluationMetrics metrics;
    std::vector<bool> gtMatched(groundTruth.size(), false);

    for (const auto& detPoint : detected) {
        float minDist = threshold;
        int matchIdx = -1;

        // Find the closest unmatched ground truth ball
        for (size_t j = 0; j < groundTruth.size(); ++j) {
            if (gtMatched[j]) continue;

            float dist = cv::norm(detPoint - groundTruth[j]);
            if (dist < minDist) {
                minDist = dist;
                matchIdx = (int)j;
            }
        }

        if (matchIdx != -1) {
            // It successfully aligned with a real ball
            metrics.TP++;
            gtMatched[matchIdx] = true;
        }
        else {
            // A detection that matches nothing is a hallucination (False Positive)
            metrics.FP++;
        }
    }

    // Any ground truth ball that wasn't found is a missed ball (False Negative)
    for (bool matched : gtMatched) {
        if (!matched) metrics.FN++;
    }

    // Calculate final metrics
    if (metrics.TP + metrics.FP > 0)
        metrics.precision = (double)metrics.TP / (metrics.TP + metrics.FP);

    if (metrics.TP + metrics.FN > 0)
        metrics.recall = (double)metrics.TP / (metrics.TP + metrics.FN);

    if (metrics.precision + metrics.recall > 0)
        metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);

    return metrics;
}


// Required entry point function
void MyApplication() {
    std::cout << "=== Pool Ball System Initialized ===" << std::endl;

    // List of all 6 test tables to process automatically
    std::vector<std::string> tableFilenames = {
        "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
        "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"
    };

    // Accumulators for overall dataset evaluation
    int totalTP = 0;
    int totalFP = 0;
    int totalFN = 0;
    int totalGroundTruthCount = 0;
    int totalDetectedCount = 0;

    for (const std::string& filename : tableFilenames) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing " << filename << "..." << std::endl;

        cv::Mat img = cv::imread(filename);
        if (img.empty()) {
            std::cerr << "Error: Could not load " << filename << "! Skipping." << std::endl;
            continue;
        }

        // 1. Warp to flat top-down view
        cv::Mat flatTable = warpTable(img);

        // 2. Run detection pipeline (this shows the windows and returns the coordinate vector)
        std::vector<cv::Point2f> detectedCenters = detectBalls(flatTable, img);
        totalDetectedCount += detectedCenters.size();

        // 3. Save the resulting image automatically to disk
        cv::Mat resultOutput = renderResults(flatTable, detectedCenters);
        std::string outputImageName = "Result_" + filename;
        cv::imwrite(outputImageName, resultOutput);
        std::cout << "Saved visual output as: " << outputImageName << std::endl;

        // 4. Evaluate against ground truth map
        if (allGroundTruths.find(filename) != allGroundTruths.end()) {
            std::vector<cv::Point2f> groundTruth = allGroundTruths[filename];
            totalGroundTruthCount += groundTruth.size();

            // Evaluate with a 30.0f pixel tolerance
            EvaluationMetrics results = evaluateDetections(detectedCenters, groundTruth, 30.0f);

            // Accumulate totals for overall metrics
            totalTP += results.TP;
            totalFP += results.FP;
            totalFN += results.FN;

            // Print individual table results
            std::cout << "--- Scorecard for " << filename << " ---" << std::endl;
            std::cout << "  * Detected vs GT Count: " << detectedCenters.size() << " / " << groundTruth.size() << std::endl;
            std::cout << "  * True Positives:       " << results.TP << std::endl;
            std::cout << "  * False Positives:      " << results.FP << std::endl;
            std::cout << "  * False Negatives:      " << results.FN << std::endl;
            std::cout << "  * Precision:            " << results.precision * 100.0 << "%" << std::endl;
            std::cout << "  * Recall:               " << results.recall * 100.0 << "%" << std::endl;
            std::cout << "  * F1-Score:             " << results.f1_score * 100.0 << "%" << std::endl;
        }
        else {
            std::cerr << "Warning: No ground truth data found for " << filename << std::endl;
        }
    }

    // 5. Calculate Overall Dataset Metrics (Micro-averaged across all 6 images)
    double overallPrecision = (totalTP + totalFP > 0) ? (double)totalTP / (totalTP + totalFP) : 0.0;
    double overallRecall = (totalTP + totalFN > 0) ? (double)totalTP / (totalTP + totalFN) : 0.0;
    double overallF1 = (overallPrecision + overallRecall > 0) ?
        2.0 * (overallPrecision * overallRecall) / (overallPrecision + overallRecall) : 0.0;

    std::cout << "\n==============================================" << std::endl;
    std::cout << "        OVERALL DATASET EVALUATION (1-6)      " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "  * Total True Positives:  " << totalTP << std::endl;
    std::cout << "  * Total False Positives: " << totalFP << std::endl;
    std::cout << "  * Total False Negatives: " << totalFN << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "  * Overall Precision: " << overallPrecision * 100.0 << "%" << std::endl;
    std::cout << "  * Overall Recall:    " << overallRecall * 100.0 << "%" << std::endl;
    std::cout << "  * Overall F1-Score:  " << overallF1 * 100.0 << "%" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    std::cout << "Batch execution finished successfully! Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}


int main() {
    MyApplication();
    return 0;
}