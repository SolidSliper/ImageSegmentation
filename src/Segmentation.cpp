
#include "Segmentation.h"

#include <opencv2/imgproc.hpp>

#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>

// Anonymous namespace - pomocne funkcie pristupne iba v tomto subore.
// Nahradza povodny "voľný" priestor v ImageSegmentationMath.cpp.
namespace {

// ----------------------------------------------------------------------------
// fillHolesInGraph: zaplni diery v segmentacii bez pouzitia cv::floodFill.
// Implementacia: vytvori binarnu masku FG/BG (visited=true => FG),
// potom BFS z border pixelov pre najdenie BG pixelov spojenych s okrajom.
// Pixel, ktory nie je FG a nie je spojeny s okrajom = "diera". Diery
// su spojite komponenty tychto pixelov; tie mensie nez ignoreHoleSize sa zaplnia.
// ----------------------------------------------------------------------------
void fillHolesInGraph(const std::vector<std::vector<Segmentation::Node*>>& pixelNodes,
                      std::vector<bool>& visited,
                      int m, int n,
                      int ignoreHoleSize)
{
    std::vector<std::vector<char>> mask(m, std::vector<char>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Segmentation::Node* node = pixelNodes[i][j];
            mask[i][j] = visited[node->id] ? 1 : 0;
        }
    }

    std::vector<std::vector<char>> bgConnected(m, std::vector<char>(n, 0));
    std::queue<std::pair<int, int>> q;
    for (int i = 0; i < m; ++i) {
        if (mask[i][0] == 0) { bgConnected[i][0] = 1; q.push({ i, 0 }); }
        if (n > 1 && mask[i][n - 1] == 0) { bgConnected[i][n - 1] = 1; q.push({ i, n - 1 }); }
    }
    for (int j = 0; j < n; ++j) {
        if (mask[0][j] == 0) { bgConnected[0][j] = 1; q.push({ 0, j }); }
        if (m > 1 && mask[m - 1][j] == 0) { bgConnected[m - 1][j] = 1; q.push({ m - 1, j }); }
    }

    int dr[4] = { -1, 1, 0, 0 };
    int dc[4] = { 0, 0, -1, 1 };
    while (!q.empty()) {
        auto p = q.front(); q.pop();
        int r = p.first, c = p.second;
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                if (!bgConnected[nr][nc] && mask[nr][nc] == 0) {
                    bgConnected[nr][nc] = 1;
                    q.push({ nr, nc });
                }
            }
        }
    }

    if (ignoreHoleSize <= 0) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mask[i][j] == 0 && !bgConnected[i][j]) {
                    visited[pixelNodes[i][j]->id] = true;
                }
            }
        }
        return;
    }

    std::vector<std::vector<char>> seen(m, std::vector<char>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (mask[i][j] == 0 && !bgConnected[i][j] && !seen[i][j]) {
                std::vector<std::pair<int, int>> comp;
                std::queue<std::pair<int, int>> q2;
                q2.push({ i, j }); seen[i][j] = 1;
                while (!q2.empty()) {
                    auto pp = q2.front(); q2.pop();
                    comp.push_back(pp);
                    int r = pp.first, c = pp.second;
                    for (int d = 0; d < 4; ++d) {
                        int nr = r + dr[d];
                        int nc = c + dc[d];
                        if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                            if (!seen[nr][nc] && mask[nr][nc] == 0 && !bgConnected[nr][nc]) {
                                seen[nr][nc] = 1;
                                q2.push({ nr, nc });
                            }
                        }
                    }
                }
                if (static_cast<int>(comp.size()) < ignoreHoleSize) {
                    for (auto& pos : comp) {
                        visited[pixelNodes[pos.first][pos.second]->id] = true;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// removeNoiseFromGraph: odstrani male skupiny pixelov objektu zo segmentacie
// analyzovanim spojitych komponent v 2D poli pixelov.
// ----------------------------------------------------------------------------
void removeNoiseFromGraph(const std::vector<std::vector<Segmentation::Node*>>& pixelNodes,
                          std::vector<bool>& visited,
                          int threshold)
{
    int m = static_cast<int>(pixelNodes.size());
    if (m == 0) return;
    int n = static_cast<int>(pixelNodes[0].size());

    std::vector<std::vector<bool>> seen(m, std::vector<bool>(n, false));
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
                if (component.size() < static_cast<size_t>(threshold)) {
                    for (const auto& pos : component) {
                        visited[pixelNodes[pos.first][pos.second]->id] = false;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// computeConvexHull: vypocet convex hull pomocou "monotone chain" algoritmu.
// ----------------------------------------------------------------------------
std::vector<cv::Point> computeConvexHull(std::vector<cv::Point> pts)
{
    if (pts.size() <= 1) return pts;
    std::sort(pts.begin(), pts.end(), [](const cv::Point& a, const cv::Point& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });
    std::vector<cv::Point> lower, upper;
    for (const auto& pt : pts) {
        while (lower.size() >= 2 &&
            ((lower[lower.size() - 1] - lower[lower.size() - 2]).cross(
                cv::Point(pt.x - lower[lower.size() - 2].x,
                          pt.y - lower[lower.size() - 2].y)) <= 0))
            lower.pop_back();
        lower.push_back(pt);
    }
    for (int i = static_cast<int>(pts.size()) - 1; i >= 0; i--) {
        while (upper.size() >= 2 &&
            ((upper[upper.size() - 1] - upper[upper.size() - 2]).cross(
                cv::Point(pts[i].x - upper[upper.size() - 2].x,
                          pts[i].y - upper[upper.size() - 2].y)) <= 0))
            upper.pop_back();
        upper.push_back(pts[i]);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

// ----------------------------------------------------------------------------
// polygonArea: shoelace vzorec
// ----------------------------------------------------------------------------
double polygonArea(const std::vector<cv::Point>& poly)
{
    double area = 0.0;
    int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; i++) {
        cv::Point p = poly[i];
        cv::Point q = poly[(i + 1) % n];
        area += (p.x * q.y - q.x * p.y);
    }
    return std::abs(area) / 2.0;
}

// ----------------------------------------------------------------------------
// Pomocna struktura pre priebezny stav konstrukcie grafu.
// ----------------------------------------------------------------------------
struct GraphData {
    Segmentation::Graph* graph;
    std::vector<std::vector<Segmentation::Node*>> pixelNodes;
    int m, n;
    Segmentation::Mode actualMode;
};

// ----------------------------------------------------------------------------
// createGraphData: vytvori sieť pre segmentaciu z obrazu.
// ----------------------------------------------------------------------------
GraphData createGraphData(const cv::Mat& inputImage,
                          double lambda,
                          Segmentation::Mode mode,
                          int userObjectIntensity,
                          int userBackgroundIntensity,
                          const cv::Mat& roiMask)
{
    GraphData data;
    data.m = inputImage.rows;
    data.n = inputImage.cols;
    data.graph = new Segmentation::Graph();
    data.graph->source = new Segmentation::Node(0);
    data.graph->sink   = new Segmentation::Node(1);
    data.graph->nodes.push_back(data.graph->source);
    data.graph->nodes.push_back(data.graph->sink);

    double minVal, maxVal;
    cv::minMaxLoc(inputImage, &minVal, &maxVal);
    int tol = std::max(15, static_cast<int>((maxVal - minVal) * 0.2));

    // Auto-detekcia rezimu
    if (mode == Segmentation::Mode::Auto) {
        int countLight = 0, countDark = 0;
        for (int i = 0; i < inputImage.rows; i++) {
            for (int j = 0; j < inputImage.cols; j++) {
                int intensity = inputImage.at<uchar>(i, j);
                if (roiMask.empty() || roiMask.at<uchar>(i, j) != 0) {
                    if (intensity >= maxVal - tol) countLight++;
                    if (intensity <= minVal + tol) countDark++;
                }
            }
        }
        mode = (countDark < countLight) ? Segmentation::Mode::Dark
                                        : Segmentation::Mode::Light;
    }
    data.actualMode = mode;

    int objectSeed, backgroundSeed;
    if (data.actualMode == Segmentation::Mode::Light) {
        objectSeed     = static_cast<int>(maxVal);
        backgroundSeed = static_cast<int>(minVal);
    } else {
        objectSeed     = static_cast<int>(minVal);
        backgroundSeed = static_cast<int>(maxVal);
    }

    // Vytvor pixelove uzly
    int id = 2;
    data.pixelNodes.resize(data.m, std::vector<Segmentation::Node*>(data.n, nullptr));
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            int intensity = inputImage.at<uchar>(i, j);
            auto* p = new Segmentation::Node(id++, intensity, j, i);
            data.pixelNodes[i][j] = p;
            data.graph->nodes.push_back(p);
        }
    }

    // T-linky (source/sink)
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            auto* p = data.pixelNodes[i][j];
            int intensity = p->intensity;
            double capSource = 0.0, capSink = 0.0;

            if (!roiMask.empty()) {
                if (roiMask.at<uchar>(i, j) == 0) {
                    capSource = 0.0;
                    capSink   = std::numeric_limits<double>::max();
                }
            }
            if (capSource == 0.0 && capSink == 0.0) {
                bool isLight = (data.actualMode == Segmentation::Mode::Light);
                if (isLight) {
                    if (intensity >= objectSeed - tol) {
                        capSource = std::numeric_limits<double>::max(); capSink = 0;
                    } else if (intensity <= backgroundSeed + tol) {
                        capSource = 0; capSink = std::numeric_limits<double>::max();
                    } else {
                        double Rs = (maxVal - minVal) - std::abs(userObjectIntensity - intensity);
                        double Rt = (maxVal - minVal) - std::abs(userBackgroundIntensity - intensity);
                        capSource = lambda * Rs;
                        capSink   = lambda * Rt;
                    }
                } else {
                    if (intensity <= objectSeed + tol) {
                        capSource = std::numeric_limits<double>::max(); capSink = 0;
                    } else if (intensity >= backgroundSeed - tol) {
                        capSource = 0; capSink = std::numeric_limits<double>::max();
                    } else {
                        double Rs = (maxVal - minVal) - std::abs(userObjectIntensity - intensity);
                        double Rt = (maxVal - minVal) - std::abs(userBackgroundIntensity - intensity);
                        capSource = lambda * Rs;
                        capSink   = lambda * Rt;
                    }
                }
            }

            auto* edge1  = new Segmentation::Link(capSource, p);
            auto* redge1 = new Segmentation::Link(0, data.graph->source);
            edge1->reverse  = redge1;
            redge1->reverse = edge1;
            data.graph->source->adj.push_back(edge1);
            p->adj.push_back(redge1);

            auto* edge2  = new Segmentation::Link(capSink, data.graph->sink);
            auto* redge2 = new Segmentation::Link(0, p);
            edge2->reverse  = redge2;
            redge2->reverse = edge2;
            p->adj.push_back(edge2);
            data.graph->sink->adj.push_back(redge2);
        }
    }

    // N-linky (4-suseda)
    for (int i = 0; i < data.m; i++) {
        for (int j = 0; j < data.n; j++) {
            auto* p = data.pixelNodes[i][j];
            if (j < data.n - 1) {
                auto* q = data.pixelNodes[i][j + 1];
                int diff = std::abs(p->intensity - q->intensity);
                double cap = (diff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - diff));
                auto* edge  = new Segmentation::Link(cap, q);
                auto* redge = new Segmentation::Link(cap, p);
                edge->reverse  = redge;
                redge->reverse = edge;
                p->adj.push_back(edge);
                q->adj.push_back(redge);
            }
            if (i < data.m - 1) {
                auto* q = data.pixelNodes[i + 1][j];
                int diff = std::abs(p->intensity - q->intensity);
                double cap = (diff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - diff));
                auto* edge  = new Segmentation::Link(cap, q);
                auto* redge = new Segmentation::Link(cap, p);
                edge->reverse  = redge;
                redge->reverse = edge;
                p->adj.push_back(edge);
                q->adj.push_back(redge);
            }
        }
    }
    return data;
}

// ----------------------------------------------------------------------------
// Dinic max-flow algoritmus
// ----------------------------------------------------------------------------
bool dinicBFS(Segmentation::Graph* graph, std::vector<int>& level)
{
    std::fill(level.begin(), level.end(), -1);
    std::queue<Segmentation::Node*> q;
    level[graph->source->id] = 0;
    q.push(graph->source);

    while (!q.empty()) {
        auto* u = q.front(); q.pop();
        for (auto edge : u->adj) {
            if (level[edge->to->id] < 0 && edge->flow < edge->capacity) {
                level[edge->to->id] = level[u->id] + 1;
                q.push(edge->to);
            }
        }
    }
    return (level[graph->sink->id] >= 0);
}

double sendFlow(Segmentation::Node* u,
                double flow,
                Segmentation::Node* sink,
                std::vector<int>& level,
                std::vector<int>& start)
{
    if (u == sink) return flow;

    for (; start[u->id] < static_cast<int>(u->adj.size()); start[u->id]++) {
        auto* edge = u->adj[start[u->id]];
        if (level[edge->to->id] == level[u->id] + 1 && edge->flow < edge->capacity) {
            double currFlow = std::min(flow, edge->capacity - edge->flow);
            double tempFlow = sendFlow(edge->to, currFlow, sink, level, start);
            if (tempFlow > 0) {
                edge->flow          += tempFlow;
                edge->reverse->flow -= tempFlow;
                return tempFlow;
            }
        }
    }
    return 0;
}

double maxFlowDinic(Segmentation::Graph* graph, int maxNodeId)
{
    double flow = 0;
    std::vector<int> level(maxNodeId, -1);
    while (dinicBFS(graph, level)) {
        std::vector<int> start(maxNodeId, 0);
        while (double currFlow = sendFlow(graph->source,
                                          std::numeric_limits<double>::max(),
                                          graph->sink, level, start)) {
            flow += currFlow;
        }
    }
    return flow;
}

// ----------------------------------------------------------------------------
// Edmonds-Karp max-flow algoritmus
// ----------------------------------------------------------------------------
const double EPS = 1e-9;

bool bfsEdmondsKarp(Segmentation::Graph* graph,
                    int maxNodeId,
                    std::vector<Segmentation::Link*>& parentEdge)
{
    std::fill(parentEdge.begin(), parentEdge.end(), nullptr);
    std::queue<Segmentation::Node*> q;
    std::vector<char> visited(maxNodeId, 0);

    q.push(graph->source);
    visited[graph->source->id] = 1;

    while (!q.empty()) {
        Segmentation::Node* u = q.front(); q.pop();
        for (Segmentation::Link* e : u->adj) {
            Segmentation::Node* v = e->to;
            double residual = e->capacity - e->flow;
            if (!visited[v->id] && residual > EPS) {
                parentEdge[v->id] = e;
                visited[v->id] = 1;
                if (v == graph->sink) return true;
                q.push(v);
            }
        }
    }
    return false;
}

double maxFlowEdmondsKarp(Segmentation::Graph* graph, int maxNodeId)
{
    double maxFlow = 0.0;
    std::vector<Segmentation::Link*> parentEdge(maxNodeId, nullptr);

    while (bfsEdmondsKarp(graph, maxNodeId, parentEdge)) {
        double pathFlow = std::numeric_limits<double>::infinity();
        Segmentation::Node* v = graph->sink;
        while (v != graph->source) {
            Segmentation::Link* e = parentEdge[v->id];
            if (e == nullptr) { pathFlow = 0.0; break; }
            double residual = e->capacity - e->flow;
            pathFlow = std::min(pathFlow, residual);
            v = e->reverse->to;
        }
        if (pathFlow <= EPS || pathFlow == std::numeric_limits<double>::infinity())
            break;

        v = graph->sink;
        while (v != graph->source) {
            Segmentation::Link* e   = parentEdge[v->id];
            Segmentation::Link* rev = e->reverse;
            e->flow   += pathFlow;
            rev->flow -= pathFlow;
            v = rev->to;
        }
        maxFlow += pathFlow;
    }
    return maxFlow;
}

// ----------------------------------------------------------------------------
// getSegmentationCut: BFS po hranach s volnou kapacitou - oznaci S-stranu rezu.
// ----------------------------------------------------------------------------
void getSegmentationCut(Segmentation::Graph* graph, std::vector<bool>& visited, int maxNodeId)
{
    visited.assign(maxNodeId, false);
    std::queue<Segmentation::Node*> q;
    q.push(graph->source);
    visited[graph->source->id] = true;

    while (!q.empty()) {
        auto* u = q.front(); q.pop();
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

// ----------------------------------------------------------------------------
// createOutputImages: farebne vizualizacie segmentu (objekt + hrany).
// ----------------------------------------------------------------------------
void createOutputImages(const cv::Mat& inputImage,
                        const std::vector<std::vector<Segmentation::Node*>>& pixelNodes,
                        const std::vector<bool>& visited,
                        cv::Mat& outputObject,
                        cv::Mat& outputEdge)
{
    cv::Mat colorImage;
    cv::cvtColor(inputImage, colorImage, cv::COLOR_GRAY2BGR);
    outputObject = colorImage.clone();
    outputEdge   = colorImage.clone();

    int m = inputImage.rows, n = inputImage.cols;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto* node = pixelNodes[i][j];
            if (visited[node->id])
                outputObject.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            else
                outputObject.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto* node = pixelNodes[i][j];
            if (visited[node->id]) {
                bool isEdge = false;
                if (i > 0     && !visited[pixelNodes[i - 1][j]->id]) isEdge = true;
                if (i < m - 1 && !visited[pixelNodes[i + 1][j]->id]) isEdge = true;
                if (j > 0     && !visited[pixelNodes[i][j - 1]->id]) isEdge = true;
                if (j < n - 1 && !visited[pixelNodes[i][j + 1]->id]) isEdge = true;
                if (isEdge)
                    outputEdge.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Verejne API triedy Segmentation
// ============================================================================

bool Segmentation::segmentImage(const cv::Mat& input,
                                const Params& params,
                                Result& result,
                                const cv::Mat& roiMask)
{
    if (input.empty()) return false;

    // Vytvor sieť
    GraphData data = createGraphData(input,
                                     params.lambda,
                                     params.mode,
                                     params.userObject,
                                     params.userBackground,
                                     roiMask);

    int maxNodeId = 2 + input.rows * input.cols;

    // Spusti zvoleny max-flow algoritmus
    if (params.algorithm == Algorithm::Dinic)
        maxFlowDinic(data.graph, maxNodeId);
    else
        maxFlowEdmondsKarp(data.graph, maxNodeId);

    // Ziskaj rez, vypln diery, odstran sum
    std::vector<bool> visited(maxNodeId, false);
    getSegmentationCut(data.graph, visited, maxNodeId);
    fillHolesInGraph(data.pixelNodes, visited, data.m, data.n, params.holeSize);
    removeNoiseFromGraph(data.pixelNodes, visited, params.noiseThreshold);

    // Vytvor vystupne obrazy
    createOutputImages(input, data.pixelNodes, visited,
                       result.objectImage, result.edgeImage);

    // Binarna maska
    result.objectMask = cv::Mat::zeros(input.size(), CV_8UC1);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            if (visited[data.pixelNodes[i][j]->id])
                result.objectMask.at<uchar>(i, j) = 255;
        }
    }

    result.isLightObject = (data.actualMode == Mode::Light);

    delete data.graph;
    return true;
}

void Segmentation::computeSeedIntensities(const cv::Mat& inputImage,
                                          Mode mode,
                                          int& objectIntensity,
                                          int& backgroundIntensity,
                                          const cv::Mat& roiMask)
{
    double minVal = 255, maxVal = 0;
    int tol = 15;
    int countPixels = 0;
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            if (!roiMask.empty() && roiMask.at<uchar>(i, j) == 0) continue;
            int intensity = inputImage.at<uchar>(i, j);
            minVal = std::min(minVal, static_cast<double>(intensity));
            maxVal = std::max(maxVal, static_cast<double>(intensity));
            countPixels++;
        }
    }
    if (countPixels == 0) cv::minMaxLoc(inputImage, &minVal, &maxVal);
    tol = std::max(15, static_cast<int>((maxVal - minVal) * 0.2));

    int objectSeed, backgroundSeed;
    if (mode == Mode::Light) {
        objectSeed     = static_cast<int>(maxVal);
        backgroundSeed = static_cast<int>(minVal);
    } else {
        objectSeed     = static_cast<int>(minVal);
        backgroundSeed = static_cast<int>(maxVal);
    }

    int sumObj = 0, countObj = 0;
    int sumBack = 0, countBack = 0;
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            if (!roiMask.empty() && roiMask.at<uchar>(i, j) == 0) continue;
            int intensity = inputImage.at<uchar>(i, j);
            if (mode == Mode::Light) {
                if (intensity >= objectSeed     - tol) { sumObj  += intensity; countObj++;  }
                if (intensity <= backgroundSeed + tol) { sumBack += intensity; countBack++; }
            } else {
                if (intensity <= objectSeed     + tol) { sumObj  += intensity; countObj++;  }
                if (intensity >= backgroundSeed - tol) { sumBack += intensity; countBack++; }
            }
        }
    }
    objectIntensity     = (countObj  > 0 ? sumObj  / countObj  : objectSeed);
    backgroundIntensity = (countBack > 0 ? sumBack / countBack : backgroundSeed);
}

// ============================================================================
// Geometricke parametre
// ============================================================================

void Segmentation::computeFeretDiameterAndCircle(const cv::Mat& input,
                                                 const cv::Mat& objectMask,
                                                 cv::Mat& annotatedImage,
                                                 double& longestFeret,
                                                 double& circleDiameter,
                                                 double& shortestFeret)
{
    if (input.channels() == 1)
        cv::cvtColor(input, annotatedImage, cv::COLOR_GRAY2BGR);
    else
        annotatedImage = input.clone();

    longestFeret   = 0.0;
    shortestFeret  = std::numeric_limits<double>::max();
    circleDiameter = 0.0;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(objectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return;

    double maxArea = 0.0;
    int maxIdx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxIdx = static_cast<int>(i); }
    }
    std::vector<cv::Point> largest = contours[maxIdx];
    std::vector<cv::Point> hull = computeConvexHull(largest);

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

    cv::Point2f bestNormal(0, 0);
    shortestFeret = std::numeric_limits<double>::max();
    for (size_t i = 0; i < hull.size(); i++) {
        cv::Point2f p1 = hull[i];
        cv::Point2f p2 = hull[(i + 1) % hull.size()];
        cv::Point2f edge = p2 - p1;
        double edgeLen = cv::norm(edge);
        if (edgeLen == 0) continue;
        cv::Point2f normal(-edge.y / edgeLen, edge.x / edgeLen);
        double minProj =  std::numeric_limits<double>::max();
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

    cv::Moments mu = cv::moments(largest);
    cv::Point2f centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);

    cv::Point2f shortPt1 = centroid - (bestNormal * (shortestFeret / 2.0f));
    cv::Point2f shortPt2 = centroid + (bestNormal * (shortestFeret / 2.0f));
    cv::line(annotatedImage, shortPt1, shortPt2, cv::Scalar(255, 0, 0), 3);

    double areaVal = polygonArea(largest);
    circleDiameter = 2.0 * std::sqrt(areaVal / CV_PI);
    int radius = static_cast<int>(circleDiameter / 2.0);
    cv::circle(annotatedImage, centroid, radius, cv::Scalar(255, 0, 0), 2);

    cv::polylines(annotatedImage, largest, true, cv::Scalar(0, 255, 255), 2);
}

void Segmentation::computeLegendreEllipse(const cv::Mat& input,
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
        if (area > maxArea) { maxArea = area; maxIdx = static_cast<int>(i); }
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

    cv::Point2f majorVec( std::cos(theta),  std::sin(theta));
    cv::Point2f minorVec(-std::sin(theta),  std::cos(theta));
    cv::Point ptMajor1 = cv::Point(cx, cy) - cv::Point(majorVec * static_cast<float>(majorAxis / 2));
    cv::Point ptMajor2 = cv::Point(cx, cy) + cv::Point(majorVec * static_cast<float>(majorAxis / 2));
    cv::Point ptMinor1 = cv::Point(cx, cy) - cv::Point(minorVec * static_cast<float>(minorAxis / 2));
    cv::Point ptMinor2 = cv::Point(cx, cy) + cv::Point(minorVec * static_cast<float>(minorAxis / 2));
    cv::line(annotatedImage, ptMajor1, ptMajor2, cv::Scalar(0, 0, 255), 2);
    cv::line(annotatedImage, ptMinor1, ptMinor2, cv::Scalar(255, 0, 0), 2);
}

void Segmentation::computeMBR(const cv::Mat& input,
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
        if (a > maxArea) { maxArea = a; maxIdx = static_cast<int>(i); }
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
        double minX =  std::numeric_limits<double>::max(),
               minY =  std::numeric_limits<double>::max();
        double maxX = -std::numeric_limits<double>::max(),
               maxY = -std::numeric_limits<double>::max();
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
    longDiameter  = std::max(d1, d2);
    shortDiameter = std::min(d1, d2);

    cv::Point2f mid0 = (bestRect[0] + bestRect[1]) * 0.5f;
    cv::Point2f mid1 = (bestRect[1] + bestRect[2]) * 0.5f;
    cv::Point2f mid2 = (bestRect[2] + bestRect[3]) * 0.5f;
    cv::Point2f mid3 = (bestRect[3] + bestRect[0]) * 0.5f;
    cv::line(annotatedImage, mid0, mid2, cv::Scalar(255, 255, 0), 2);
    cv::line(annotatedImage, mid1, mid3, cv::Scalar(255, 255, 0), 2);
}
