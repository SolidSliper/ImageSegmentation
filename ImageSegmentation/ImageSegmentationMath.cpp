#include "ImageSegmentation.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------
// Vnorene triedy grafu pouzite v algoritme segmentacie.
class ImageSegmentation::Node {
public:
    int id;
    int intensity; // pre pixelove uzly
    int x, y;      // suradnice pixelu (alebo -1 pre source/sink)
    std::vector<Link*> adj;
    Node(int id, int intensity = 0, int x = -1, int y = -1)
        : id(id), intensity(intensity), x(x), y(y) {
    }
};

class ImageSegmentation::Link {
public:
    double capacity;
    double flow;
    Node* to;
    Link* reverse;
    Link(double cap, Node* toNode)
        : capacity(cap), flow(0), to(toNode), reverse(nullptr) {
    }
};

class ImageSegmentation::Graph {
public:
    std::vector<Node*> nodes;
    Node* source;
    Node* sink;
    ~Graph() {
        for (Node* node : nodes) {
            for (Link* edge : node->adj)
                delete edge;
            delete node;
        }
    }
};

// ---------------------------------------------------------------------
// Pomocne funkcie pre anotaciu obrazka a matematicke spracovanie.

// annotateImage: Anotuje obrazok s popisom oblasti.
cv::Mat ImageSegmentation::annotateImage(const cv::Mat& img, double areaObject, int areaImage)
{
    cv::Mat annotated;
    if (img.channels() == 1)
        cv::cvtColor(img, annotated, cv::COLOR_GRAY2BGR);
    else
        annotated = img.clone();

    std::string text = "Area: " + std::to_string(areaObject) + " (" +
        std::to_string((areaImage > 0) ? (areaObject * 100 / areaImage) : 0) + "%)";
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    int margin = 10;
    cv::Point textOrg(annotated.cols - textSize.width - margin, annotated.rows - margin);
    cv::putText(annotated, text, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
    cv::putText(annotated, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    return annotated;
}

// fillHolesInGraph: Zaplni diery v segmentacii objektu pomocou flood fill.
void fillHolesInGraph(const std::vector<std::vector<ImageSegmentation::Node*>>& pixelNodes,
    std::vector<bool>& visited, int m, int n)
{
    cv::Mat mask = cv::Mat::zeros(m, n, CV_8UC1);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ImageSegmentation::Node* node = pixelNodes[i][j];
            if (visited[node->id])
                mask.at<uchar>(i, j) = 255;
        }
    }
    cv::Mat flood = mask.clone();
    for (int i = 0; i < m; i++) {
        if (flood.at<uchar>(i, 0) == 0)
            cv::floodFill(flood, cv::Point(0, i), cv::Scalar(128));
        if (flood.at<uchar>(i, n - 1) == 0)
            cv::floodFill(flood, cv::Point(n - 1, i), cv::Scalar(128));
    }
    for (int j = 0; j < n; j++) {
        if (flood.at<uchar>(0, j) == 0)
            cv::floodFill(flood, cv::Point(j, 0), cv::Scalar(128));
        if (flood.at<uchar>(m - 1, j) == 0)
            cv::floodFill(flood, cv::Point(j, m - 1), cv::Scalar(128));
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (flood.at<uchar>(i, j) == 0) {
                ImageSegmentation::Node* node = pixelNodes[i][j];
                visited[node->id] = true;
            }
        }
    }
}

// removeNoiseFromGraph: Odstrani male skupiny pixelov objektu zo segmentacie
// analyzovanim spojitych komponent v 2D poli pixelov.
void removeNoiseFromGraph(const std::vector<std::vector<ImageSegmentation::Node*>>& pixelNodes,
    std::vector<bool>& visited,
    int threshold)
{
    int m = pixelNodes.size();
    if (m == 0)
        return;
    int n = pixelNodes[0].size();

    // 2D pole na oznacenie uz spracovanych pixelov
    std::vector<std::vector<bool>> seen(m, std::vector<bool>(n, false));

    // 4-smerove susedne posuny
    int dr[4] = { -1, 1, 0, 0 };
    int dc[4] = { 0, 0, -1, 1 };

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (visited[pixelNodes[i][j]->id] && !seen[i][j]) {
                std::queue<std::pair<int, int>> q;
                std::vector<std::pair<int, int>> component;

                q.push({ i, j });
                seen[i][j] = true;

                while (!q.empty()) {
                    auto [r, c] = q.front();
                    q.pop();
                    component.push_back({ r, c });

                    for (int d = 0; d < 4; d++) {
                        int nr = r + dr[d];
                        int nc = c + dc[d];
                        if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                            if (visited[pixelNodes[nr][nc]->id] && !seen[nr][nc]) {
                                seen[nr][nc] = true;
                                q.push({ nr, nc });
                            }
                        }
                    }
                }
                // ak je komponent mensi nez threshold, nastav ho na pozadie
                if (component.size() < static_cast<size_t>(threshold)) {
                    for (const auto& pos : component) {
                        int r = pos.first;
                        int c = pos.second;
                        visited[pixelNodes[r][c]->id] = false;
                    }
                }
            }
        }
    }
}

// computeConvexHull: Vypocet convex hull pomocou "monotone chain" algoritmu.
std::vector<cv::Point> computeConvexHull(std::vector<cv::Point> pts)
{
    if (pts.size() <= 1)
        return pts;
    std::sort(pts.begin(), pts.end(), [](const cv::Point& a, const cv::Point& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
        });
    std::vector<cv::Point> lower, upper;
    for (const auto& pt : pts) {
        while (lower.size() >= 2 &&
            ((lower[lower.size() - 1] - lower[lower.size() - 2]).cross(
                cv::Point(pt.x - lower[lower.size() - 2].x, pt.y - lower[lower.size() - 2].y)) <= 0))
            lower.pop_back();
        lower.push_back(pt);
    }
    for (int i = pts.size() - 1; i >= 0; i--) {
        while (upper.size() >= 2 &&
            ((upper[upper.size() - 1] - upper[upper.size() - 2]).cross(
                cv::Point(pts[i].x - upper[upper.size() - 2].x, pts[i].y - upper[upper.size() - 2].y)) <= 0))
            upper.pop_back();
        upper.push_back(pts[i]);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

// polygonArea: Vypocet plochy mnohouholnika pomocou shoelace vzorca.
double polygonArea(const std::vector<cv::Point>& poly)
{
    double area = 0.0;
    int n = poly.size();
    for (int i = 0; i < n; i++) {
        cv::Point p = poly[i];
        cv::Point q = poly[(i + 1) % n];
        area += (p.x * q.y - q.x * p.y);
    }
    return std::abs(area) / 2.0;
}

// ---------------------------------------------------------------------
// Implementacia matematikych metod ImageSegmentation.

// computeFeretDiameterAndCircle: Vypocita Feretove priemery a priemer kruhu ekvivalentneho ploche.
void ImageSegmentation::computeFeretDiameterAndCircle(const cv::Mat& input,
    const cv::Mat& objectMask,
    cv::Mat& annotatedImage,
    double& longestFeret,
    double& circleDiameter,
    double& shortestFeret)
{
    // priprava obrazka na kreslenie
    if (input.channels() == 1)
        cv::cvtColor(input, annotatedImage, cv::COLOR_GRAY2BGR);
    else
        annotatedImage = input.clone();

    longestFeret = 0.0;
    shortestFeret = std::numeric_limits<double>::max();
    circleDiameter = 0.0;

    // najdeme najvacsiu konturu
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(objectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return;

    double maxArea = 0.0;
    int maxIdx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = i;
        }
    }
    std::vector<cv::Point> largest = contours[maxIdx];
    std::vector<cv::Point> hull = computeConvexHull(largest);

    // vypocet najdlhsieho Feret priemeru
    cv::Point bestLongP1, bestLongP2;
    double bestLongDist = 0.0;
    for (size_t i = 0; i < hull.size(); i++) {
        for (size_t j = i + 1; j < hull.size(); j++) {
            double d = cv::norm(hull[i] - hull[j]);
            if (d > bestLongDist) {
                bestLongDist = d;
                bestLongP1 = hull[i];
                bestLongP2 = hull[j];
            }
        }
    }
    longestFeret = bestLongDist;
    cv::line(annotatedImage, bestLongP1, bestLongP2, cv::Scalar(0, 0, 255), 3);

    // vypocet najkratsieho Feret priemeru
    cv::Point2f bestNormal(0, 0);
    shortestFeret = std::numeric_limits<double>::max();
    for (size_t i = 0; i < hull.size(); i++) {
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()];
        cv::Point2f edge = p2 - p1;
        double edgeLen = cv::norm(edge);
        if (edgeLen == 0) continue;
        cv::Point2f normal(-edge.y / edgeLen, edge.x / edgeLen);
        double minProj = std::numeric_limits<double>::max();
        double maxProj = -std::numeric_limits<double>::max();
        for (const auto& pt : hull) {
            double proj = pt.dot(normal);
            minProj = std::min(minProj, proj);
            maxProj = std::max(maxProj, proj);
        }
        double width = maxProj - minProj;
        if (width < shortestFeret) {
            shortestFeret = width;
            bestNormal = normal;
        }
    }

    // vypocet stredu hmotnosti
    cv::Moments mu = cv::moments(largest);
    cv::Point2f centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);

    // kreslenie najkratsieho priemeru
    cv::Point2f shortPt1 = centroid - (bestNormal * (shortestFeret / 2.0f));
    cv::Point2f shortPt2 = centroid + (bestNormal * (shortestFeret / 2.0f));
    cv::line(annotatedImage, shortPt1, shortPt2, cv::Scalar(255, 0, 0), 3);

    // vypocet priemeru ekvivaletnej kruznice
    double areaVal = polygonArea(largest);
    circleDiameter = 2.0 * std::sqrt(areaVal / CV_PI);
    int radius = static_cast<int>(circleDiameter / 2.0);
    cv::circle(annotatedImage, centroid, radius, cv::Scalar(255, 0, 0), 2);

    // prekreslime obrys objektu
    cv::polylines(annotatedImage, largest, true, cv::Scalar(0, 255, 255), 2);
}

// computeLegendreEllipse: vypocita parametre elipsy z momentov obrazu.
void ImageSegmentation::computeLegendreEllipse(const cv::Mat& input,
    const cv::Mat& objectMask,
    cv::Mat& annotatedImage,
    double& majorAxis,
    double& minorAxis)
{
    if (input.channels() == 1)
        cv::cvtColor(input, annotatedImage, cv::COLOR_GRAY2BGR);
    else
        annotatedImage = input.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(objectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { majorAxis = minorAxis = 0; return; }

    double maxArea = 0;
    int maxIdx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxIdx = i; }
    }
    std::vector<cv::Point> contour = contours[maxIdx];
    if (contour.size() < 5) { majorAxis = minorAxis = 0; return; }

    double m00 = 0, m10 = 0, m01 = 0;
    double mu20 = 0, mu02 = 0, mu11 = 0;
    for (const auto& pt : contour) {
        m00 += 1;
        m10 += pt.x;
        m01 += pt.y;
    }
    if (m00 == 0) { majorAxis = minorAxis = 0; return; }

    double cx = m10 / m00;
    double cy = m01 / m00;
    for (const auto& pt : contour) {
        double dx = pt.x - cx;
        double dy = pt.y - cy;
        mu20 += dx * dx;
        mu02 += dy * dy;
        mu11 += dx * dy;
    }
    double common = std::sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11);
    double lambda1 = (mu20 + mu02 + common) / m00;
    double lambda2 = (mu20 + mu02 - common) / m00;
    majorAxis = 2 * std::sqrt(lambda1);
    minorAxis = 2 * std::sqrt(lambda2);
    double theta = 0.5 * std::atan2(2 * mu11, (mu20 - mu02));
    double angleDeg = theta * 180.0 / CV_PI;
    cv::ellipse(annotatedImage, cv::Point(cx, cy),
        cv::Size(static_cast<int>(majorAxis / 2), static_cast<int>(minorAxis / 2)),
        angleDeg, 0, 360, cv::Scalar(0, 165, 255), 2);

    cv::Point2f majorVec(std::cos(theta), std::sin(theta));
    cv::Point2f minorVec(-std::sin(theta), std::cos(theta));
    cv::Point ptMajor1 = cv::Point(cx, cy) - cv::Point(majorVec * static_cast<float>(majorAxis / 2));
    cv::Point ptMajor2 = cv::Point(cx, cy) + cv::Point(majorVec * static_cast<float>(majorAxis / 2));
    cv::Point ptMinor1 = cv::Point(cx, cy) - cv::Point(minorVec * static_cast<float>(minorAxis / 2));
    cv::Point ptMinor2 = cv::Point(cx, cy) + cv::Point(minorVec * static_cast<float>(minorAxis / 2));
    cv::line(annotatedImage, ptMajor1, ptMajor2, cv::Scalar(0, 0, 255), 2);
    cv::line(annotatedImage, ptMinor1, ptMinor2, cv::Scalar(255, 0, 0), 2);
}

// computeMBR: vypocita minimalny ohranny obdlznik pomocou rotujucich kaliprov.
void ImageSegmentation::computeMBR(const cv::Mat& input,
    const cv::Mat& objectMask,
    cv::Mat& annotatedImage,
    double& longDiameter,
    double& shortDiameter)
{
    if (input.channels() == 1)
        cv::cvtColor(input, annotatedImage, cv::COLOR_GRAY2BGR);
    else
        annotatedImage = input.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(objectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { longDiameter = shortDiameter = 0; return; }
    double maxArea = 0;
    int maxIdx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) { maxArea = a; maxIdx = i; }
    }
    std::vector<cv::Point> hull = computeConvexHull(contours[maxIdx]);
    if (hull.size() < 3) { longDiameter = shortDiameter = 0; return; }
    double minAreaRect = std::numeric_limits<double>::max();
    std::vector<cv::Point2f> bestRect;
    for (size_t i = 0; i < hull.size(); i++) {
        cv::Point2f p0 = hull[i];
        cv::Point2f p1 = hull[(i + 1) % hull.size()];
        double angle = std::atan2(p1.y - p0.y, p1.x - p0.x);
        double cosA = std::cos(-angle), sinA = std::sin(-angle);
        double minX = std::numeric_limits<double>::max(), minY = std::numeric_limits<double>::max();
        double maxX = -std::numeric_limits<double>::max(), maxY = -std::numeric_limits<double>::max();
        for (const auto& pt : hull) {
            double rx = pt.x * cosA - pt.y * sinA;
            double ry = pt.x * sinA + pt.y * cosA;
            minX = std::min(minX, rx);
            minY = std::min(minY, ry);
            maxX = std::max(maxX, rx);
            maxY = std::max(maxY, ry);
        }
        double areaRect = (maxX - minX) * (maxY - minY);
        if (areaRect < minAreaRect) {
            minAreaRect = areaRect;
            std::vector<cv::Point2f> rect;
            rect.push_back(cv::Point2f(minX, minY));
            rect.push_back(cv::Point2f(maxX, minY));
            rect.push_back(cv::Point2f(maxX, maxY));
            rect.push_back(cv::Point2f(minX, maxY));
            bestRect.clear();
            double cosInv = std::cos(angle), sinInv = std::sin(angle);
            for (auto& pt : rect) {
                float x = pt.x * cosInv - pt.y * sinInv;
                float y = pt.x * sinInv + pt.y * cosInv;
                bestRect.push_back(cv::Point2f(x, y));
            }
        }
    }
    if (bestRect.empty()) { longDiameter = shortDiameter = 0; return; }
    for (int i = 0; i < 4; i++)
        cv::line(annotatedImage, bestRect[i], bestRect[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    double d1 = cv::norm(bestRect[0] - bestRect[1]);
    double d2 = cv::norm(bestRect[1] - bestRect[2]);
    longDiameter = std::max(d1, d2);
    shortDiameter = std::min(d1, d2);
    cv::Point2f mid0 = (bestRect[0] + bestRect[1]) * 0.5f;
    cv::Point2f mid1 = (bestRect[1] + bestRect[2]) * 0.5f;
    cv::Point2f mid2 = (bestRect[2] + bestRect[3]) * 0.5f;
    cv::Point2f mid3 = (bestRect[3] + bestRect[0]) * 0.5f;
    cv::line(annotatedImage, mid0, mid2, cv::Scalar(255, 255, 0), 2);
    cv::line(annotatedImage, mid1, mid3, cv::Scalar(255, 255, 0), 2);
}

// computeSeedIntensities: urci intenzity seed bodov pre objekt a pozadie.
void ImageSegmentation::computeSeedIntensities(const cv::Mat& inputImage,
    ImageSegmentation::SegmentationMode mode,
    int& defaultObjectIntensity,
    int& defaultBackgroundIntensity,
    const cv::Mat& roiMask)
{
    double minVal = 255, maxVal = 0;
    int tol = 15;
    int countPixels = 0;
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            if (!roiMask.empty() && roiMask.at<uchar>(i, j) == 0)
                continue;
            int intensity = inputImage.at<uchar>(i, j);
            minVal = std::min(minVal, static_cast<double>(intensity));
            maxVal = std::max(maxVal, static_cast<double>(intensity));
            countPixels++;
        }
    }
    if (countPixels == 0) cv::minMaxLoc(inputImage, &minVal, &maxVal);
    tol = std::max(15, static_cast<int>((maxVal - minVal) * 0.2));
    int objectSeed, backgroundSeed;
    if (mode == ImageSegmentation::SegmentationMode::Light) {
        objectSeed = static_cast<int>(maxVal);
        backgroundSeed = static_cast<int>(minVal);
    }
    else {
        objectSeed = static_cast<int>(minVal);
        backgroundSeed = static_cast<int>(maxVal);
    }
    int sumObj = 0, countObj = 0;
    int sumBack = 0, countBack = 0;
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            if (!roiMask.empty() && roiMask.at<uchar>(i, j) == 0)
                continue;
            int intensity = inputImage.at<uchar>(i, j);
            if (mode == ImageSegmentation::SegmentationMode::Light) {
                if (intensity >= objectSeed - tol) { sumObj += intensity; countObj++; }
                if (intensity <= backgroundSeed + tol) { sumBack += intensity; countBack++; }
            }
            else {
                if (intensity <= objectSeed + tol) { sumObj += intensity; countObj++; }
                if (intensity >= backgroundSeed - tol) { sumBack += intensity; countBack++; }
            }
        }
    }
    defaultObjectIntensity = (countObj > 0 ? sumObj / countObj : objectSeed);
    defaultBackgroundIntensity = (countBack > 0 ? sumBack / countBack : backgroundSeed);
}

// ---------------------------------------------------------------------
// konstrukcia grafu a max flow (Dinicov algoritmus) pomocne funkcie.
struct GraphData {
    ImageSegmentation::Graph* graph;
    std::vector<std::vector<ImageSegmentation::Node*>> pixelNodes;
    int m, n;
    ImageSegmentation::SegmentationMode actualMode;
};

GraphData createGraphData(const cv::Mat& inputImage,
    double lambda,
    ImageSegmentation::SegmentationMode mode,
    bool overrideSeeds,
    int userObjectIntensity,
    int userBackgroundIntensity,
    const cv::Mat& roiMask)
{
    GraphData data;
    data.m = inputImage.rows;
    data.n = inputImage.cols;
    data.graph = new ImageSegmentation::Graph();
    data.graph->source = new ImageSegmentation::Node(0);
    data.graph->sink = new ImageSegmentation::Node(1);
    data.graph->nodes.push_back(data.graph->source);
    data.graph->nodes.push_back(data.graph->sink);
    double minVal, maxVal;
    cv::minMaxLoc(inputImage, &minVal, &maxVal);
    int tol = std::max(15, static_cast<int>((maxVal - minVal) * 0.2));
    if (mode == ImageSegmentation::SegmentationMode::Auto) {
        int count_light = 0, count_dark = 0;
        for (int i = 0; i < inputImage.rows; i++) {
            for (int j = 0; j < inputImage.cols; j++) {
                int intensity = inputImage.at<uchar>(i, j);
                if (roiMask.empty() || roiMask.at<uchar>(i, j) != 0) {
                    if (intensity >= maxVal - tol) count_light++;
                    if (intensity <= minVal + tol) count_dark++;
                }
            }
        }
        mode = (count_dark < count_light) ? ImageSegmentation::SegmentationMode::Dark : ImageSegmentation::SegmentationMode::Light;
        data.actualMode = mode;
    }
    else {
        data.actualMode = mode;
    }
    int objectSeed, backgroundSeed;
    if (data.actualMode == ImageSegmentation::SegmentationMode::Light) {
        objectSeed = static_cast<int>(maxVal);
        backgroundSeed = static_cast<int>(minVal);
    }
    else {
        objectSeed = static_cast<int>(minVal);
        backgroundSeed = static_cast<int>(maxVal);
    }
    int id = 2;
    data.pixelNodes.resize(data.m, std::vector<ImageSegmentation::Node*>(data.n, nullptr));
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            int intensity = inputImage.at<uchar>(i, j);
            auto* p = new ImageSegmentation::Node(id++, intensity, j, i);
            data.pixelNodes[i][j] = p;
            data.graph->nodes.push_back(p);
        }
    }
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            auto* p = data.pixelNodes[i][j];
            int intensity = p->intensity;
            double capSource = 0.0, capSink = 0.0;
            if (!roiMask.empty()) {
                if (roiMask.at<uchar>(i, j) == 0) {
                    capSource = 0.0;
                    capSink = std::numeric_limits<double>::max();
                }
            }
            if (capSource == 0.0 && capSink == 0.0) {
                bool isLight = (data.actualMode == ImageSegmentation::SegmentationMode::Light);
                if (isLight) {
                    if (intensity >= objectSeed - tol) { capSource = std::numeric_limits<double>::max(); capSink = 0; }
                    else if (intensity <= backgroundSeed + tol) { capSource = 0; capSink = std::numeric_limits<double>::max(); }
                    else {
                        double Rs = (maxVal - minVal) - std::abs(userObjectIntensity - intensity);
                        double Rt = (maxVal - minVal) - std::abs(userBackgroundIntensity - intensity);
                        capSource = lambda * Rs;
                        capSink = lambda * Rt;
                    }
                }
                else {
                    if (intensity <= objectSeed + tol) { capSource = std::numeric_limits<double>::max(); capSink = 0; }
                    else if (intensity >= backgroundSeed - tol) { capSource = 0; capSink = std::numeric_limits<double>::max(); }
                    else {
                        double Rs = (maxVal - minVal) - std::abs(userObjectIntensity - intensity);
                        double Rt = (maxVal - minVal) - std::abs(userBackgroundIntensity - intensity);
                        capSource = lambda * Rs;
                        capSink = lambda * Rt;
                    }
                }
            }
            auto* edge1 = new ImageSegmentation::Link(capSource, p);
            auto* redge1 = new ImageSegmentation::Link(0, data.graph->source);
            edge1->reverse = redge1;
            redge1->reverse = edge1;
            data.graph->source->adj.push_back(edge1);
            p->adj.push_back(redge1);
            auto* edge2 = new ImageSegmentation::Link(capSink, data.graph->sink);
            auto* redge2 = new ImageSegmentation::Link(0, p);
            edge2->reverse = redge2;
            redge2->reverse = edge2;
            p->adj.push_back(edge2);
            data.graph->sink->adj.push_back(redge2);
        }
    }
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            auto* p = data.pixelNodes[i][j];
            if (j < data.n - 1) {
                auto* q = data.pixelNodes[i][j + 1];
                int diff = std::abs(p->intensity - q->intensity);
                double cap = (diff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - diff));
                auto* edge = new ImageSegmentation::Link(cap, q);
                auto* redge = new ImageSegmentation::Link(cap, p);
                edge->reverse = redge;
                redge->reverse = edge;
                p->adj.push_back(edge);
                q->adj.push_back(redge);
            }
            if (i < data.m - 1) {
                auto* q = data.pixelNodes[i + 1][j];
                int diff = std::abs(p->intensity - q->intensity);
                double cap = (diff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - diff));
                auto* edge = new ImageSegmentation::Link(cap, q);
                auto* redge = new ImageSegmentation::Link(cap, p);
                edge->reverse = redge;
                redge->reverse = edge;
                p->adj.push_back(edge);
                q->adj.push_back(redge);
            }
        }
    }
    return data;
}

// dinicBFS: naplanuje vrstvu (level) pre kazdy uzol v grafe pomoci BFS.
// Vrstva nam hovori, kolko hran je na najkratsiej ceste od zdroja.
// Vrstva sa pouzije na urcenie, ktore hrany mozu byt sucastou dalsieho augmentacneho toku.
bool dinicBFS(ImageSegmentation::Graph* graph, std::vector<int>& level, int maxNodeId)
{
    // Inicializuj vsetky vrstvy na -1 (nenavstivene)
    std::fill(level.begin(), level.end(), -1);

    // Kvoli BFS pouzijeme frontu uzlov
    std::queue<ImageSegmentation::Node*> q;

    // Vrstva zdroja je 0
    level[graph->source->id] = 0;
    q.push(graph->source);

    // BFS prechadza vsetky hrany, kde je kapacita > pruden
    while (!q.empty()) {
        auto* u = q.front();
        q.pop();

        // Pre kazdu hranu z u
        for (auto edge : u->adj) {
            // Ak cielovy uzol este nie je navstiveny a hrana ma volnu kapacitu
            if (level[edge->to->id] < 0 && edge->flow < edge->capacity) {
                // Nastav vrstvu a pridaj uzol do fronty
                level[edge->to->id] = level[u->id] + 1;
                q.push(edge->to);
            }
        }
    }

    // Ak sme dosiahli drez, vrati true, inak false
    return (level[graph->sink->id] >= 0);
}

// sendFlow: posle co najvacsi mozny tok od uzla u po vrstvach definovanych v level.
// Rekurzivne skusa prehladat cesty z u do sink a vrati vyslany tok.
double sendFlow(ImageSegmentation::Node* u,
    double flow,
    ImageSegmentation::Node* sink,
    std::vector<int>& level,
    std::vector<int>& start)
{
    // Ak sme dosiahli sink, vratime aktualny mozny tok
    if (u == sink)
        return flow;

    // Pre kazdu hranu vychadzajucu z u, od zaciatku definovaneho v start[u->id]
    for (; start[u->id] < static_cast<int>(u->adj.size()); start[u->id]++) {
        auto* edge = u->adj[start[u->id]];

        // Podmienka: hrana musi ist do uzla o vrstvu dalej a mat volnu kapacitu
        if (level[edge->to->id] == level[u->id] + 1 && edge->flow < edge->capacity) {
            // maximalny tok, ktory mozeme poslat touto hranou
            double curr_flow = std::min(flow, edge->capacity - edge->flow);

            // rekurzivne posli tok dalej
            double temp_flow = sendFlow(edge->to, curr_flow, sink, level, start);

            // ak sa podarilo poslat nejaky tok, update hranu a reverznu hranu
            if (temp_flow > 0) {
                edge->flow += temp_flow;
                edge->reverse->flow -= temp_flow;
                return temp_flow;
            }
        }
    }

    // ak ziadny tok nemoze ist dalej, vrat 0
    return 0;
}

// maxFlowDinic: hlavna funkcia, co spusti Dinicov algoritmus.
// Strieda vytvorenie vrstiev BFS a posielanie blokovaneho toku, kym sa da.
double maxFlowDinic(ImageSegmentation::Graph* graph, int maxNodeId)
{
    double flow = 0;
    // Pole pre ulozenie vrstiev pre vsetky uzly
    std::vector<int> level(maxNodeId, -1);

    // Dokedy existuje cesta od zdroja do drezu (vrstvy su definovane)
    while (dinicBFS(graph, level, maxNodeId)) {
        // Pole start urcuje, od ktorej hrany pokracovat pre kazdy uzol
        std::vector<int> start(maxNodeId, 0);

        // Pokial mozeme poslat nejaky tok, inkrementuj celkovy flow
        while (double curr_flow = sendFlow(graph->source, std::numeric_limits<double>::max(), graph->sink, level, start))
        {
            qDebug() << curr_flow;
            flow += curr_flow;
        }
        // Po vycerpani blokovaneho toku preco BFS znova obnovi vrstvy
    }

    return flow;
}

const double EPS = 1e-9;
const double INF_CAP = 1e12; // alebo dynamicky spocitane hodnoty podla obrazu

// BFS that fills parentEdge: parentEdge[v->id] = edge used to reach v (edge->to == v)
bool bfsEdmondsKarp(ImageSegmentation::Graph* graph,
    int maxNodeId,
    std::vector<ImageSegmentation::Link*>& parentEdge)
{
    std::fill(parentEdge.begin(), parentEdge.end(), nullptr);
    std::queue<ImageSegmentation::Node*> q;
    std::vector<char> visited(maxNodeId, 0);

    q.push(graph->source);
    visited[graph->source->id] = 1;

    while (!q.empty()) {
        ImageSegmentation::Node* u = q.front(); q.pop();
        for (ImageSegmentation::Link* e : u->adj) {
            ImageSegmentation::Node* v = e->to;
            double residual = e->capacity - e->flow;
            if (!visited[v->id] && residual > EPS) {
                parentEdge[v->id] = e;     // we reached v via edge e (u -> v)
                visited[v->id] = 1;
                if (v == graph->sink) {
                    return true; // found path to sink
                }
                q.push(v);
            }
        }
    }
    return false; // no augmenting path
}

double maxFlowEdmondsKarp(ImageSegmentation::Graph* graph, int maxNodeId)
{
    double maxFlow = 0.0;

    // parentEdge[v_id] = edge used to reach v in BFS
    std::vector<ImageSegmentation::Link*> parentEdge(maxNodeId, nullptr);

    // Repeat while there exists an augmenting path (found by BFS)
    while (bfsEdmondsKarp(graph, maxNodeId, parentEdge)) {
        // find bottleneck (minimum residual) along the path sink <- ... <- source
        double path_flow = std::numeric_limits<double>::infinity();
        ImageSegmentation::Node* v = graph->sink;

        while (v != graph->source) {
            ImageSegmentation::Link* e = parentEdge[v->id];
            if (e == nullptr) { // defensive - should not happen if BFS returned true
                path_flow = 0.0;
                break;
            }
            double residual = e->capacity - e->flow;
            path_flow = std::min(path_flow, residual);
            // prev node is edge->reverse->to (because reverse->to points to 'from')
            v = e->reverse->to;
        }

        if (path_flow <= EPS || path_flow == std::numeric_limits<double>::infinity()) {
            // nothing to push (defensive). Should not normally happen.
            break;
        }

        // augment along the path: increase forward flow, decrease reverse flow
        v = graph->sink;
        while (v != graph->source) {
            ImageSegmentation::Link* e = parentEdge[v->id];
            ImageSegmentation::Link* rev = e->reverse;
            e->flow += path_flow;
            rev->flow -= path_flow;
            v = rev->to; // move to previous node
        }

        maxFlow += path_flow;
    }

    return maxFlow;
}



// getSegmentationCut: po vykonani max flow vyuzije residualny graf
// a BFS/DFS zisti, ktore uzly su dosiahnutelne zo zdroja.
// Navstivene uzly tvoria S-stranu rezov a definuju segmentovane pixely.
void getSegmentationCut(ImageSegmentation::Graph* graph, std::vector<bool>& visited, int maxNodeId)
{
    // Inicializuj navstivene na false
    visited.assign(maxNodeId, false);

    std::queue<ImageSegmentation::Node*> q;
    q.push(graph->source);
    visited[graph->source->id] = true;

    // BFS po hranach, kde zostala volna kapacita (capacity - flow > 0)
    while (!q.empty()) {
        auto* u = q.front();
        q.pop();

        for (auto edge : u->adj) {
            if (edge->capacity - edge->flow > 0) {
                auto* v = edge->to;
                if (!visited[v->id]) {
                    visited[v->id] = true;
                    q.push(v);
                }
            }
        }
    }
}

// createOutputImages: vygeneruje dve farebne vizualizacie vyslednej segmentacie.
// outputObject: cele oblasti objektu vyfarbene cervene.
// outputEdge: iba hranice objektu cervene.
void createOutputImages(const cv::Mat& inputImage,
    const std::vector<std::vector<ImageSegmentation::Node*>>& pixelNodes,
    const std::vector<bool>& visited,
    cv::Mat& outputObject,
    cv::Mat& outputEdge)
{
    // Konvertuj vstup na BGR pre farebne kreslenie
    cv::Mat colorImage;
    cv::cvtColor(inputImage, colorImage, cv::COLOR_GRAY2BGR);
    outputObject = colorImage.clone();
    outputEdge = colorImage.clone();

    int m = inputImage.rows, n = inputImage.cols;

    // Vykresli vsetky pixely, ktore su v S-strane (visited = true) cervene
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto* node = pixelNodes[i][j];
            if (visited[node->id]) {
                // objekt
                outputObject.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    // Detekuj hranice: pixel je hranou, ak ma aspon jedneho suseda nie v S-strane
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto* node = pixelNodes[i][j];
            if (visited[node->id]) {
                bool isEdge = false;
                if (i > 0 && !visited[pixelNodes[i - 1][j]->id]) isEdge = true;
                if (i < m - 1 && !visited[pixelNodes[i + 1][j]->id]) isEdge = true;
                if (j > 0 && !visited[pixelNodes[i][j - 1]->id]) isEdge = true;
                if (j < n - 1 && !visited[pixelNodes[i][j + 1]->id]) isEdge = true;
                if (isEdge) {
                    outputEdge.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                }
            }
        }
    }
}


// updateSeedIntensities: prepocita intenzity seed bodov pomocou ROI masky a aktualizuje ovladace.
void ImageSegmentation::updateSeedIntensities()
{
    if (inputImage.empty())
        return;

    // Ak uzivatel nakreslil ROI, pouzi tuto masku; inak ziadna maska.
    const cv::Mat& mask = userROIMask.empty() ? cv::Mat() : userROIMask;

    // Prepocitaj seed intenzity na celej obrazku, ale filtrovanie iba pixlov v maske (ak je).
    computeSeedIntensities(
        inputImage,
        currentMode,
        defaultObjectIntensity,
        defaultBackgroundIntensity,
        mask
    );

    // Nastav nove hodnoty seed intenzit do UI ovladacov
    ui.doubleSpinBoxObject->setValue(defaultObjectIntensity);
    ui.doubleSpinBoxBackground->setValue(defaultBackgroundIntensity);
}

// segmentImage: hlavna funkcia segmentacie, ktora stavia graf, spusta max flow,
// ziska cut, postprocessuje masku a vrati vysledky.
bool ImageSegmentation::segmentImage(const cv::Mat& input,
    SegmentationMode mode,
    double lambda,
    cv::Mat& outputObject,
    cv::Mat& outputEdges,
    cv::Mat& objectMask,
    bool& isLightObject,
    const cv::Mat& roiMask)
{
    // Prepocitaj seed intenzity na vstupe
    computeSeedIntensities(input, mode, defaultObjectIntensity, defaultBackgroundIntensity, roiMask);

    // Vytvor data pre graf a spust max flow algoritmus
    GraphData data = createGraphData(input, lambda, mode, true, userObject, userBackground, roiMask);
    int maxNodeId = 2 + input.rows * input.cols;
    if(ui.comboBoxAlgorithm->currentText() == "Dinic")
        double flow = maxFlowDinic(data.graph, maxNodeId);
    else
        double flow = maxFlowEdmondsKarp(data.graph, maxNodeId);

    // Ziskaj Rez, vypln diery a odstran sum
    std::vector<bool> visited(maxNodeId, false);
    getSegmentationCut(data.graph, visited, maxNodeId);

    fillHolesInGraph(data.pixelNodes, visited, data.m, data.n);
    removeNoiseFromGraph(data.pixelNodes, visited, threshold);

    // Vytvor vystupne obrazky pre objekt a hrany
    createOutputImages(input, data.pixelNodes, visited, outputObject, outputEdges);

    // Vygeneruj binary masku objektu
    objectMask = cv::Mat::zeros(input.size(), CV_8UC1);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (visited[data.pixelNodes[i][j]->id])
                objectMask.at<uchar>(i, j) = 255;
        }
    }

    // Morphologia: open a close pre zjemnenie masky
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(objectMask, objectMask, cv::MORPH_OPEN, kernelOpen);
    cv::morphologyEx(objectMask, objectMask, cv::MORPH_CLOSE, kernelClose);

    // Nastav rezim a uloz poslednu masku
    isLightObject = (data.actualMode == SegmentationMode::Light);
    lastObjectMask = objectMask;
    lastIsLightObject = isLightObject;

    delete data.graph;
    return true;
}
