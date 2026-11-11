#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>

// --- GROUND TRUTH STRUCTURES ---
struct GTBall {
    int id; // 0 for White, 1-15 for numbered balls
    cv::Point2f pos;
};

struct RecognizedBall {
    cv::Point2f center;
    int id; // Predicted ID (0 for White, 1-15 for numbered balls, -1 for unknown)
};

// Map of Image Filename -> Labeled Ground Truth (Ball ID + Position)
std::map<std::string, std::vector<GTBall>> labeledGroundTruths = {
    { "AllBalls.jpeg",{
        { 1,{ 908.0, 253.0 } },{ 2,{ 959.0, 250.0 } },{ 3,{ 1013.0, 255.0 } },{ 4,{ 1072.0, 256.0 } },
        { 5,{ 1132.0, 254.0 } },{ 6,{ 1184.0, 262.0 } },{ 7,{ 1244.0, 256.0 } },{ 8,{ 1307.0, 253.0 } },
        { 9,{ 901.0, 329.0 } },{ 10,{ 961.0, 330.0 } },{ 11,{ 1013.0, 335.0 } },{ 12,{ 1067.0, 336.0 } },
        { 13,{ 1130.0, 334.0 } },{ 14,{ 1187.0, 345.0 } },{ 15,{ 1245.0, 342.0 } },{ 0,{ 797.0, 293.0 } }
    } },
    { "Table1.jpeg",{
        { 1,{ 1092.0, 309.0 } },{ 2,{ 1212.0, 429.0 } },{ 3,{ 1131.0, 383.0 } },{ 4,{ 1050.0, 335.0 } },
        { 5,{ 1211.0, 285.0 } },{ 6,{ 1170.0, 261.0 } },{ 7,{ 1173.0, 352.0 } },{ 8,{ 1132.0, 331.0 } },
        { 9,{ 1210.0, 240.0 } },{ 10,{ 1170.0, 310.0 } },{ 11,{ 1169.0, 407.0 } },{ 12,{ 1135.0, 281.0 } },
        { 13,{ 1212.0, 374.0 } },{ 14,{ 1091.0, 360.0 } },{ 15,{ 1213.0, 331.0 } },{ 0,{ 29.0, 288.0 } }
    } },
    { "Table2.jpeg",{
        { 1,{ 1103.0, 282.0 } },{ 2,{ 1224.0, 620.0 } },{ 3,{ 1121.0, 391.0 } },{ 4,{ 1046.0, 315.0 } },
        { 5,{ 1305.0, 310.0 } },{ 6,{ 621.0, 202.0 } },{ 7,{ 1173.0, 361.0 } },{ 8,{ 1132.0, 331.0 } },
        { 9,{ 725.0, 177.0 } },{ 10,{ 1269.0, 185.0 } },{ 11,{ 798.0, 665.0 } },{ 12,{ 438.0, 296.0 } },
        { 13,{ 1235.0, 53.0 } },{ 14,{ 1090.0, 362.0 } },{ 15,{ 1205.0, 399.0 } },{ 0,{ 1169.0, 507.0 } }
    } },
    { "Table3.jpeg",{
        { 1,{ 908.0, 23.0 } },{ 2,{ 1224.0, 616.0 } },{ 3,{ 1199.0, 447.0 } },{ 4,{ 29.0, 336.0 } },
        { 5,{ 1406.0, 305.0 } },{ 6,{ 621.0, 210.0 } },{ 7,{ 1294.0, 260.0 } },{ 8,{ 1252.0, 226.0 } },
        { 9,{ 725.0, 177.0 } },{ 10,{ 1270.0, 171.0 } },{ 11,{ 798.0, 665.0 } },{ 12,{ 351.0, 249.0 } },
        { 13,{ 1233.0, 53.0 } },{ 14,{ 901.0, 169.0 } },{ 15,{ 1201.0, 399.0 } },{ 0,{ 29.0, 434.0 } }
    } },
    { "Table4.jpeg",{
        { 1,{ 908.0, 23.0 } },{ 2,{ 1407.0, 546.0 } },{ 3,{ 1204.0, 444.0 } },{ 5,{ 1406.0, 305.0 } },
        { 6,{ 621.0, 210.0 } },{ 7,{ 1294.0, 260.0 } },{ 8,{ 1252.0, 226.0 } },{ 9,{ 724.0, 181.0 } },
        { 10,{ 1270.0, 171.0 } },{ 11,{ 798.0, 665.0 } },{ 12,{ 351.0, 249.0 } },{ 13,{ 1231.0, 58.0 } },
        { 14,{ 901.0, 169.0 } },{ 15,{ 1201.0, 399.0 } },{ 0,{ 1392.0, 88.0 } }
    } },
    { "Table5.jpeg",{
        { 1,{ 23.0, 535.0 } },{ 2,{ 1407.0, 546.0 } },{ 3,{ 1204.0, 444.0 } },{ 5,{ 1406.0, 305.0 } },
        { 6,{ 25.0, 131.0 } },{ 7,{ 1294.0, 260.0 } },{ 8,{ 1252.0, 226.0 } },{ 9,{ 72.0, 538.0 } },
        { 10,{ 1270.0, 171.0 } },{ 11,{ 798.0, 665.0 } },{ 12,{ 351.0, 249.0 } },{ 14,{ 901.0, 169.0 } },
        { 15,{ 1201.0, 399.0 } },{ 0,{ 26.0, 608.0 } }
    } },
    { "Table6.jpeg",{
        { 1,{ 23.0, 535.0 } },{ 2,{ 1364.0, 477.0 } },{ 3,{ 1204.0, 444.0 } },{ 5,{ 1406.0, 305.0 } },
        { 6,{ 25.0, 131.0 } },{ 7,{ 1294.0, 260.0 } },{ 8,{ 1252.0, 226.0 } },{ 9,{ 72.0, 538.0 } },
        { 10,{ 1270.0, 171.0 } },{ 12,{ 351.0, 249.0 } },{ 14,{ 901.0, 169.0 } },{ 15,{ 1201.0, 399.0 } },
        { 0,{ 947.0, 629.0 } }
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

cv::Mat extractBallHistogram(const cv::Mat& flatTable, cv::Point2f center) {
    int radius = 20;
    cv::Rect roi(center.x - radius, center.y - radius, radius * 2, radius * 2);

    roi &= cv::Rect(0, 0, flatTable.cols, flatTable.rows);
    if (roi.width <= 0 || roi.height <= 0) return cv::Mat();

    cv::Mat crop = flatTable(roi);
    cv::Mat hsvCrop;
    cv::cvtColor(crop, hsvCrop, cv::COLOR_BGR2HSV);

    cv::Mat mask = cv::Mat::zeros(crop.size(), CV_8UC1);
    cv::circle(mask, cv::Point(radius, radius), radius - 2, cv::Scalar(255), cv::FILLED);

    int h_bins = 30, s_bins = 32;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    cv::Mat hist;
    cv::calcHist(&hsvCrop, 1, channels, mask, hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

std::map<int, cv::Mat> buildReferenceDictionary(const cv::Mat& originalAllBallsImg) {
    std::cout << "Building Reference Histogram Dictionary from AllBalls.jpeg..." << std::endl;
    cv::Mat flatAllBalls = warpTable(originalAllBallsImg);
    std::map<int, cv::Mat> referenceHistograms;

    for (const auto& gt : labeledGroundTruths["AllBalls.jpeg"]) {
        cv::Mat hist = extractBallHistogram(flatAllBalls, gt.pos);
        referenceHistograms[gt.id] = hist;
    }
    std::cout << "Successfully extracted " << referenceHistograms.size() << " reference histograms!" << std::endl;
    return referenceHistograms;
}

int identifyBall(const cv::Mat& flatTable, cv::Point2f center, const std::map<int, cv::Mat>& referenceHistograms) {
    cv::Mat targetHist = extractBallHistogram(flatTable, center);
    if (targetHist.empty()) return -1;

    int bestMatchID = -1;
    double bestDistance = 1.0;

    for (const auto& pair : referenceHistograms) {
        int refID = pair.first;
        const cv::Mat& refHist = pair.second;

        double distance = cv::compareHist(targetHist, refHist, cv::HISTCMP_BHATTACHARYYA);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestMatchID = refID;
        }
    }
    return bestMatchID;
}

cv::Mat createEdgeMask(const cv::Mat& flatTable) {
    cv::Mat gray, blurred, edges;
    cv::cvtColor(flatTable, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, blurred, 5);
    cv::Canny(blurred, edges, 30, 100);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6)));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    return edges;
}

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

static std::vector<cv::Point2f> splitCluster(const cv::Mat& original, const cv::Mat& H,
    const cv::Mat& clusterMask, const cv::Mat& foreground, double ballArea) {
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

    std::vector<cv::Vec4f> circles = findCirclesMultiChannel(original(roi));
    if (circles.empty()) return {};

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

std::vector<cv::Point2f> extractBlobCenters(const cv::Mat& edges, const cv::Mat& original, const cv::Mat& H) {
    const int W = edges.cols, Ht = edges.rows;
    const int BAND = 45;
    const double ballArea = 3200.0;

    cv::Mat work = removeCushionLines(edges);
    cv::morphologyEx(work, work, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

    const std::vector<cv::Point> pockets = {
        { 0, 0 },{ W / 2, 0 },{ W, 0 },{ 0, Ht },{ W / 2, Ht },{ W, Ht } };
    for (size_t i = 0; i < pockets.size(); i++)
        cv::circle(work, pockets[i], (i == 1 || i == 4) ? 60 : 90, cv::Scalar(0), cv::FILLED);

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

    cv::Mat foreground;
    cv::dilate(filled, foreground, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

    cv::morphologyEx(filled, filled, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
    cv::imshow("Debug: Cleaned Filled Blobs", filled);

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

    std::vector<cv::Point2f> unique;
    for (const auto& p : centres) {
        bool dup = false;
        for (const auto& q : unique)
            if (cv::norm(p - q) < 25.0) { dup = true; break; }
        if (!dup) unique.push_back(p);
    }
    return unique;
}

cv::Mat renderResults(const cv::Mat& flatTable, const std::vector<RecognizedBall>& balls) {
    cv::Mat display = flatTable.clone();

    std::vector<cv::Point> pockets = {
        cv::Point(0, 0), cv::Point(717, 0), cv::Point(1434, 0),
        cv::Point(0, 683), cv::Point(717, 683), cv::Point(1434, 683)
    };

    int validCount = 0;
    for (const auto& ball : balls) {
        bool isPocket = false;
        for (const auto& pocket : pockets) {
            if (cv::norm(ball.center - cv::Point2f(pocket)) < 65.0) {
                isPocket = true;
                break;
            }
        }
        if (isPocket) continue;

        cv::circle(display, ball.center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(display, ball.center, 21, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        std::string label = (ball.id == 0) ? "W" : std::to_string(ball.id);
        if (ball.id == -1) label = "?";

        cv::putText(display, label, cv::Point(ball.center.x - 10, ball.center.y - 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(display, label, cv::Point(ball.center.x - 10, ball.center.y - 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        validCount++;
    }
    return display;
}

std::vector<RecognizedBall> detectBalls(const cv::Mat& flatTable, const cv::Mat& original, const std::map<int, cv::Mat>& refDict) {
    cv::Mat edges = createEdgeMask(flatTable);
    std::vector<cv::Point2f> centres = extractBlobCenters(edges, original, tableHomography());

    std::vector<RecognizedBall> recognizedBalls;
    for (const auto& c : centres) {
        int predictedID = identifyBall(flatTable, c, refDict);
        recognizedBalls.push_back({ c, predictedID });
    }

    cv::Mat finalOutput = renderResults(flatTable, recognizedBalls);
    cv::imshow("Detected & Identified Balls", finalOutput);

    return recognizedBalls;
}

struct EvaluationMetrics {
    // Stage 1 (Detection)
    int TP = 0, FP = 0, FN = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1_score = 0.0;

    // Stage 2 (Identification)
    int correctID = 0;
    int incorrectID = 0;
    double accuracyMatched = 0.0;  // Correct IDs / True Positives
    double accuracyOverall = 0.0;  // Correct IDs / Total Ground Truth
};

EvaluationMetrics evaluateDetections(const std::vector<RecognizedBall>& detected,
    const std::vector<GTBall>& groundTruth,
    float threshold = 30.0f) {
    EvaluationMetrics metrics;
    std::vector<bool> gtMatched(groundTruth.size(), false);

    for (const auto& det : detected) {
        float minDist = threshold;
        int matchIdx = -1;

        for (size_t j = 0; j < groundTruth.size(); ++j) {
            if (gtMatched[j]) continue;

            float dist = cv::norm(det.center - groundTruth[j].pos);
            if (dist < minDist) {
                minDist = dist;
                matchIdx = (int)j;
            }
        }

        if (matchIdx != -1) {
            metrics.TP++;
            gtMatched[matchIdx] = true;

            // STAGE 2 EVALUATION: Compare predicted ID with ground truth ID
            if (det.id == groundTruth[matchIdx].id) {
                metrics.correctID++;
            }
            else {
                metrics.incorrectID++;
            }
        }
        else {
            metrics.FP++;
        }
    }

    for (bool matched : gtMatched) {
        if (!matched) metrics.FN++;
    }

    // Stage 1 Math
    if (metrics.TP + metrics.FP > 0)
        metrics.precision = (double)metrics.TP / (metrics.TP + metrics.FP);
    if (metrics.TP + metrics.FN > 0)
        metrics.recall = (double)metrics.TP / (metrics.TP + metrics.FN);
    if (metrics.precision + metrics.recall > 0)
        metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);

    // Stage 2 Math
    if (metrics.TP > 0)
        metrics.accuracyMatched = (double)metrics.correctID / metrics.TP;
    if (!groundTruth.empty())
        metrics.accuracyOverall = (double)metrics.correctID / groundTruth.size();

    return metrics;
}

void MyApplication() {
    std::cout << "=== Pool Ball Detection & Identification System ===" << std::endl;

    std::vector<std::string> tableFilenames = {
        "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
        "Table4.jpeg", "Table5.jpeg", "Table6.jpeg"
    };

    int totalTP = 0, totalFP = 0, totalFN = 0;
    int totalCorrectID = 0, totalIncorrectID = 0;
    int totalGTCount = 0;

    cv::Mat allBallsImg = cv::imread("AllBalls.jpeg");
    if (allBallsImg.empty()) {
        std::cerr << "Error: Cannot load training image AllBalls.jpeg!" << std::endl;
        return;
    }
    std::map<int, cv::Mat> referenceDict = buildReferenceDictionary(allBallsImg);

    for (const std::string& filename : tableFilenames) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Processing " << filename << "..." << std::endl;

        cv::Mat img = cv::imread(filename);
        if (img.empty()) {
            std::cerr << "Error: Could not load " << filename << "! Skipping." << std::endl;
            continue;
        }

        cv::Mat flatTable = warpTable(img);

        // Run detection and recognition pipeline
        std::vector<RecognizedBall> recognizedBalls = detectBalls(flatTable, img, referenceDict);

        // Save result image to disk
        cv::Mat resultOutput = renderResults(flatTable, recognizedBalls);
        std::string outputImageName = "Result_" + filename;
        cv::imwrite(outputImageName, resultOutput);
        std::cout << "Saved visual output as: " << outputImageName << std::endl;

        // Evaluation
        if (labeledGroundTruths.find(filename) != labeledGroundTruths.end()) {
            const auto& groundTruth = labeledGroundTruths[filename];
            totalGTCount += groundTruth.size();

            EvaluationMetrics results = evaluateDetections(recognizedBalls, groundTruth, 30.0f);

            totalTP += results.TP;
            totalFP += results.FP;
            totalFN += results.FN;
            totalCorrectID += results.correctID;
            totalIncorrectID += results.incorrectID;

            std::cout << "--- Scorecard for " << filename << " ---" << std::endl;
            std::cout << "  [Stage 1 - Detection]" << std::endl;
            std::cout << "    * Detected vs GT Count: " << recognizedBalls.size() << " / " << groundTruth.size() << std::endl;
            std::cout << "    * True Positives:       " << results.TP << std::endl;
            std::cout << "    * False Positives:      " << results.FP << std::endl;
            std::cout << "    * False Negatives:      " << results.FN << std::endl;
            std::cout << "    * Precision:            " << results.precision * 100.0 << "%" << std::endl;
            std::cout << "    * Recall:               " << results.recall * 100.0 << "%" << std::endl;
            std::cout << "    * F1-Score:             " << results.f1_score * 100.0 << "%" << std::endl;
            std::cout << "  [Stage 2 - Identification]" << std::endl;
            std::cout << "    * Correctly Identified: " << results.correctID << " / " << results.TP << " matched balls" << std::endl;
            std::cout << "    * Misidentified:        " << results.incorrectID << std::endl;
            std::cout << "    * Matched ID Accuracy:  " << results.accuracyMatched * 100.0 << "%" << std::endl;
            std::cout << "    * Overall Table Accuracy: " << results.accuracyOverall * 100.0 << "%" << std::endl;
        }
        else {
            std::cerr << "Warning: No ground truth data found for " << filename << std::endl;
        }
    }

    // Overall Dataset Metrics (Micro-averaged across all 6 tables)
    double overallPrecision = (totalTP + totalFP > 0) ? (double)totalTP / (totalTP + totalFP) : 0.0;
    double overallRecall = (totalTP + totalFN > 0) ? (double)totalTP / (totalTP + totalFN) : 0.0;
    double overallF1 = (overallPrecision + overallRecall > 0) ?
        2.0 * (overallPrecision * overallRecall) / (overallPrecision + overallRecall) : 0.0;

    double overallMatchedAccuracy = (totalTP > 0) ? (double)totalCorrectID / totalTP : 0.0;
    double overallTotalAccuracy = (totalGTCount > 0) ? (double)totalCorrectID / totalGTCount : 0.0;

    std::cout << "\n==============================================" << std::endl;
    std::cout << "        OVERALL DATASET EVALUATION (1-6)      " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "[STAGE 1: DETECTION METRICS]" << std::endl;
    std::cout << "  * Total True Positives:  " << totalTP << std::endl;
    std::cout << "  * Total False Positives: " << totalFP << std::endl;
    std::cout << "  * Total False Negatives: " << totalFN << std::endl;
    std::cout << "  * Overall Precision:     " << overallPrecision * 100.0 << "%" << std::endl;
    std::cout << "  * Overall Recall:        " << overallRecall * 100.0 << "%" << std::endl;
    std::cout << "  * Overall F1-Score:      " << overallF1 * 100.0 << "%" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "[STAGE 2: IDENTIFICATION METRICS]" << std::endl;
    std::cout << "  * Correctly Identified:  " << totalCorrectID << " / " << totalTP << " detected balls" << std::endl;
    std::cout << "  * Misidentified Balls:   " << totalIncorrectID << std::endl;
    std::cout << "  * Matched Accuracy:      " << overallMatchedAccuracy * 100.0 << "%" << std::endl;
    std::cout << "  * Full Dataset Accuracy: " << overallTotalAccuracy * 100.0 << "%" << std::endl;
    std::cout << "==============================================\n" << std::endl;

    std::cout << "Batch execution finished successfully! Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    MyApplication();
    return 0;
}