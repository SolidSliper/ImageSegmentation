#include "ImageSegmentation.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>

// =====================
// Nested Graph Classes
// =====================

class ImageSegmentation::Node {
public:
    int id;
    int intensity;   // for pixel nodes
    int x, y;        // pixel coordinates (or -1 for source/sink)
    std::vector<Link*> adj;  // outgoing edges

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
            for (Link* edge : node->adj) {
                delete edge;
            }
            delete node;
        }
    }
};

// =====================
// Helper Functions
// =====================

namespace {

    // A very large value used as “infinity.”
    const double INF = std::numeric_limits<double>::max();

    // Structure to hold our graph along with a matrix of pixel nodes.
    struct GraphData {
        ImageSegmentation::Graph* graph;
        std::vector<std::vector<ImageSegmentation::Node*>> pixelNodes;
        int m, n;
        ImageSegmentation::SegmentationMode actualMode;
    };

// ----------------------------------------------------------------
    // createGraphData()
    //   Given the input image and lambda parameter, creates the graph
    //   with source/sink, pixel nodes, and all edges.
    //   This updated version fixes dark object segmentation:
    //     - In Light mode, pixels near max intensity are forced to be object.
    //     - In Dark mode, pixels near min intensity are forced to be object.
    //     - Auto mode now chooses Dark mode if fewer dark pixels (within a tolerance)
    //       are present than light pixels.
    // ----------------------------------------------------------------
    GraphData createGraphData(const cv::Mat& inputImage, double lambda, ImageSegmentation::SegmentationMode mode) {
        GraphData data;
        data.m = inputImage.rows;
        data.n = inputImage.cols;
        data.graph = new ImageSegmentation::Graph();

        // Create source (id = 0) and sink (id = 1)
        data.graph->source = new ImageSegmentation::Node(0);
        data.graph->sink = new ImageSegmentation::Node(1);
        data.graph->nodes.push_back(data.graph->source);
        data.graph->nodes.push_back(data.graph->sink);

        // Determine minimum and maximum intensity values.
        double minVal, maxVal;
        cv::minMaxLoc(inputImage, &minVal, &maxVal);

        // Dynamic tolerance: 5% of the range, but at least 5.
        int tol = std::max(5, static_cast<int>((maxVal - minVal) * 0.05));

        // Auto mode: count pixels near the intensity extremes using the dynamic tolerance.
        if (mode == ImageSegmentation::SegmentationMode::Auto) {
            int count_light = 0, count_dark = 0;
            for (int i = 0; i < inputImage.rows; ++i) {
                for (int j = 0; j < inputImage.cols; ++j) {
                    int intensity = inputImage.at<uchar>(i, j);
                    if (intensity >= maxVal - tol)
                        count_light++;
                    if (intensity <= minVal + tol)
                        count_dark++;
                }
            }
            // If dark pixels are fewer than light, use Dark mode (object is dark); else use Light mode.
            mode = (count_dark < count_light) ? ImageSegmentation::SegmentationMode::Dark : ImageSegmentation::SegmentationMode::Light;
            data.actualMode = mode;
        } else {
            data.actualMode = mode;
        }

        // Define seed intensities based on the chosen mode.
        int objectSeed, backgroundSeed;
        if (data.actualMode == ImageSegmentation::SegmentationMode::Light) {
            objectSeed = static_cast<int>(maxVal);
            backgroundSeed = static_cast<int>(minVal);
        } else { // Dark mode: object is dark.
            objectSeed = static_cast<int>(minVal);
            backgroundSeed = static_cast<int>(maxVal);
        }

        // Compute average intensities for seed regions using the dynamic tolerance.
        int sumObj = 0, countObj = 0;
        int sumBack = 0, countBack = 0;
        if (data.actualMode == ImageSegmentation::SegmentationMode::Light) {
            for (int i = 0; i < data.m; i++) {
                for (int j = 0; j < data.n; j++) {
                    int intensity = inputImage.at<uchar>(i, j);
                    if (intensity >= objectSeed - tol) {
                        sumObj += intensity;
                        countObj++;
                    }
                    if (intensity <= backgroundSeed + tol) {
                        sumBack += intensity;
                        countBack++;
                    }
                }
            }
        } else { // Dark mode
            for (int i = 0; i < data.m; i++) {
                for (int j = 0; j < data.n; j++) {
                    int intensity = inputImage.at<uchar>(i, j);
                    if (intensity <= objectSeed + tol) {
                        sumObj += intensity;
                        countObj++;
                    }
                    if (intensity >= backgroundSeed - tol) {
                        sumBack += intensity;
                        countBack++;
                    }
                }
            }
        }
        int Is = (countObj > 0 ? sumObj / countObj : objectSeed);
        int It = (countBack > 0 ? sumBack / countBack : backgroundSeed);

        // Create pixel nodes (IDs starting from 2).
        int id = 2;
        data.pixelNodes.resize(data.m, std::vector<ImageSegmentation::Node*>(data.n, nullptr));
        for (int i = 0; i < data.m; i++) {
            for (int j = 0; j < data.n; j++) {
                int intensity = inputImage.at<uchar>(i, j);
                ImageSegmentation::Node* pixelNode = new ImageSegmentation::Node(id++, intensity, j, i);
                data.pixelNodes[i][j] = pixelNode;
                data.graph->nodes.push_back(pixelNode);
            }
        }

        // Add terminal edges (from source and to sink) for each pixel.
        for (int i = 0; i < data.m; i++) {
            for (int j = 0; j < data.n; j++) {
                ImageSegmentation::Node* p = data.pixelNodes[i][j];
                double capSource, capSink;
                if (data.actualMode == ImageSegmentation::SegmentationMode::Light) {
                    if (p->intensity >= objectSeed - tol) {
                        capSource = INF;
                        capSink = 0;
                    } else if (p->intensity <= backgroundSeed + tol) {
                        capSource = 0;
                        capSink = INF;
                    } else {
                        double Rs = (maxVal - minVal) - std::abs(Is - p->intensity);
                        double Rt = (maxVal - minVal) - std::abs(It - p->intensity);
                        capSource = lambda * Rs;
                        capSink = lambda * Rt;
                    }
                } else { // Dark mode
                    if (p->intensity <= objectSeed + tol) {
                        capSource = INF;
                        capSink = 0;
                    } else if (p->intensity >= backgroundSeed - tol) {
                        capSource = 0;
                        capSink = INF;
                    } else {
                        double Rs = (maxVal - minVal) - std::abs(Is - p->intensity);
                        double Rt = (maxVal - minVal) - std::abs(It - p->intensity);
                        capSource = lambda * Rs;
                        capSink = lambda * Rt;
                    }
                }

                // Create edge from source to pixel.
                ImageSegmentation::Link* edge1 = new ImageSegmentation::Link(capSource, p);
                ImageSegmentation::Link* redge1 = new ImageSegmentation::Link(0, data.graph->source);
                edge1->reverse = redge1;
                redge1->reverse = edge1;
                data.graph->source->adj.push_back(edge1);
                p->adj.push_back(redge1);

                // Create edge from pixel to sink.
                ImageSegmentation::Link* edge2 = new ImageSegmentation::Link(capSink, data.graph->sink);
                ImageSegmentation::Link* redge2 = new ImageSegmentation::Link(0, p);
                edge2->reverse = redge2;
                redge2->reverse = edge2;
                p->adj.push_back(edge2);
                data.graph->sink->adj.push_back(redge2);
            }
        }

        // Add edges between neighboring pixels (4-connected).
        for (int i = 0; i < data.m; i++) {
            for (int j = 0; j < data.n; j++) {
                ImageSegmentation::Node* p = data.pixelNodes[i][j];
                // Right neighbor.
                if (j < data.n - 1) {
                    ImageSegmentation::Node* q = data.pixelNodes[i][j + 1];
                    int intensityDiff = std::abs(p->intensity - q->intensity);
                    double cap = (intensityDiff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - intensityDiff));
                    ImageSegmentation::Link* edge = new ImageSegmentation::Link(cap, q);
                    ImageSegmentation::Link* redge = new ImageSegmentation::Link(cap, p);
                    edge->reverse = redge;
                    redge->reverse = edge;
                    p->adj.push_back(edge);
                    q->adj.push_back(redge);
                }
                // Down neighbor.
                if (i < data.m - 1) {
                    ImageSegmentation::Node* q = data.pixelNodes[i + 1][j];
                    int intensityDiff = std::abs(p->intensity - q->intensity);
                    double cap = (intensityDiff == 0 ? (maxVal - minVal) : ((maxVal - minVal) - intensityDiff));
                    ImageSegmentation::Link* edge = new ImageSegmentation::Link(cap, q);
                    ImageSegmentation::Link* redge = new ImageSegmentation::Link(cap, p);
                    edge->reverse = redge;
                    redge->reverse = edge;
                    p->adj.push_back(edge);
                    q->adj.push_back(redge);
                }
            }
        }
        return data;
    }

    // ----------------------------------------------------------------
    // Dinic's algorithm helper functions.
    // ----------------------------------------------------------------

    // Build level graph using BFS. Returns true if sink is reachable.
    bool dinicBFS(ImageSegmentation::Graph* graph, std::vector<int>& level, int maxNodeId) {
        std::fill(level.begin(), level.end(), -1);
        std::queue<ImageSegmentation::Node*> q;
        level[graph->source->id] = 0;
        q.push(graph->source);

        while (!q.empty()) {
            ImageSegmentation::Node* u = q.front();
            q.pop();
            for (auto edge : u->adj) {
                if (level[edge->to->id] < 0 && edge->flow < edge->capacity) {
                    level[edge->to->id] = level[u->id] + 1;
                    q.push(edge->to);
                }
            }
        }
        return (level[graph->sink->id] >= 0);
    }

    // Send flow using DFS on the level graph.
    double sendFlow(ImageSegmentation::Node* u, double flow, ImageSegmentation::Node* sink,
        std::vector<int>& level, std::vector<int>& start) {
        if (u == sink)
            return flow;

        for (; start[u->id] < u->adj.size(); start[u->id]++) {
            ImageSegmentation::Link* edge = u->adj[start[u->id]];
            if (level[edge->to->id] == level[u->id] + 1 && edge->flow < edge->capacity) {
                double curr_flow = std::min(flow, edge->capacity - edge->flow);
                double temp_flow = sendFlow(edge->to, curr_flow, sink, level, start);
                if (temp_flow > 0) {
                    edge->flow += temp_flow;
                    edge->reverse->flow -= temp_flow;
                    return temp_flow;
                }
            }
        }
        return 0;
    }

    // ----------------------------------------------------------------
    // maxFlowDinic()
    //   Computes the maximum flow on the given graph using Dinic's algorithm.
    // ----------------------------------------------------------------
    double maxFlowDinic(ImageSegmentation::Graph* graph, int maxNodeId) {
        double flow = 0;
        std::vector<int> level(maxNodeId, -1);

        while (dinicBFS(graph, level, maxNodeId)) {
            std::vector<int> start(maxNodeId, 0);
            while (double current_flow = sendFlow(graph->source, INF, graph->sink, level, start))
                flow += current_flow;
        }
        return flow;
    }

    // ----------------------------------------------------------------
    // getSegmentationCut()
    //   After computing max flow, this function does a BFS from the
    //   source in the residual graph. Nodes reached belong to the
    //   source side of the min cut.
    // ----------------------------------------------------------------
    void getSegmentationCut(ImageSegmentation::Graph* graph, std::vector<bool>& visited, int maxNodeId) {
        visited.assign(maxNodeId, false);
        std::queue<ImageSegmentation::Node*> q;
        q.push(graph->source);
        visited[graph->source->id] = true;
        while (!q.empty()) {
            ImageSegmentation::Node* u = q.front();
            q.pop();
            for (auto edge : u->adj) {
                if (edge->capacity - edge->flow > 0) {
                    ImageSegmentation::Node* v = edge->to;
                    if (!visited[v->id]) {
                        visited[v->id] = true;
                        q.push(v);
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // createOutputImages()
    //   Based on the computed min cut (given by 'visited' for each node)
    //   and the pixelNodes matrix, creates two output images:
    //     - outputObjectImage: full segmentation (object region colored red)
    //     - outputEdgeImage: only the boundary (edge pixels colored red)
    // ----------------------------------------------------------------
    void createOutputImages(const cv::Mat& inputImage,
        const std::vector<std::vector<ImageSegmentation::Node*>>& pixelNodes,
        const std::vector<bool>& visited,
        cv::Mat& outputObjectImage,
        cv::Mat& outputEdgeImage) {
        // Convert input image to a 3-channel BGR image.
        cv::Mat colorImage;
        cv::cvtColor(inputImage, colorImage, cv::COLOR_GRAY2BGR);
        outputObjectImage = colorImage.clone();
        outputEdgeImage = colorImage.clone();
        int m = inputImage.rows;
        int n = inputImage.cols;

        // Mark all pixels in the object region (visited) in red.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ImageSegmentation::Node* node = pixelNodes[i][j];
                if (visited[node->id]) {
                    outputObjectImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                }
            }
        }

        // For outputEdgeImage, mark only the boundary pixels.
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ImageSegmentation::Node* node = pixelNodes[i][j];
                if (visited[node->id]) {
                    bool isEdgePixel = false;
                    if (i > 0 && !visited[pixelNodes[i - 1][j]->id])
                        isEdgePixel = true;
                    if (i < m - 1 && !visited[pixelNodes[i + 1][j]->id])
                        isEdgePixel = true;
                    if (j > 0 && !visited[pixelNodes[i][j - 1]->id])
                        isEdgePixel = true;
                    if (j < n - 1 && !visited[pixelNodes[i][j + 1]->id])
                        isEdgePixel = true;
                    if (isEdgePixel) {
                        outputEdgeImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                    }
                }
            }
        }
    }
} // end anonymous namespace

// =====================
// Main Class Functions
// =====================
ImageSegmentation::ImageSegmentation(QWidget* parent) : QMainWindow(parent) {
    ui.setupUi(this);

    // Mode buttons
    modeButtonGroup.addButton(ui.toolButtonLight, static_cast<int>(SegmentationMode::Light));
    modeButtonGroup.addButton(ui.toolButtonDark, static_cast<int>(SegmentationMode::Dark));
    modeButtonGroup.addButton(ui.toolButtonAuto, static_cast<int>(SegmentationMode::Auto));
    connect(&modeButtonGroup, QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
        [this](QAbstractButton* button) {
            currentMode = static_cast<SegmentationMode>(modeButtonGroup.id(button));
        });
    ui.toolButtonLight->setChecked(true);

    // Connect UI signals.
    connect(ui.toolButtonObject, &QToolButton::toggled, this, &ImageSegmentation::on_toolButtonObject_toggled);
    connect(ui.toolButtonEdge, &QToolButton::toggled, this, &ImageSegmentation::on_toolButtonEdge_toggled);
    connect(ui.doubleSpinBoxLambda, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double val) { lambda = val; });
    connect(ui.doubleSpinBoxScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double val) { scaleFactor = val; });
}
      
ImageSegmentation::~ImageSegmentation() {}

bool ImageSegmentation::segmentImage(const cv::Mat& input, SegmentationMode mode, double lambda,
    cv::Mat& outputObject, cv::Mat& outputEdges, cv::Mat& objectMask, bool& isLightObject) {

    GraphData data = createGraphData(input, lambda, mode);
    int maxNodeId = 2 + input.rows * input.cols;

    // Compute max flow
    double flow = maxFlowDinic(data.graph, maxNodeId);

    // Get segmentation cut
    std::vector<bool> visited(maxNodeId, false);
    getSegmentationCut(data.graph, visited, maxNodeId);

    // Create output images and mask
    createOutputImages(input, data.pixelNodes, visited, outputObject, outputEdges);
    objectMask = cv::Mat::zeros(input.size(), CV_8UC1);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            if (visited[data.pixelNodes[i][j]->id]) {
                objectMask.at<uchar>(i, j) = 255;
            }
        }
    }

    isLightObject = (data.actualMode == SegmentationMode::Light);
    delete data.graph;
    return true;
}

void ImageSegmentation::on_actionOpen_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image", "",
        "Image Files (*.png *.jpg *.tif)");
    if (filename.isEmpty())
        return;

    inputImage = cv::imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        QMessageBox::critical(this, "Error", "Failed to load image");
        return;
    }

    // Convert the image to QImage.
    QImage qimg = cvMatToQImage(inputImage);

    // Use our helper function to display the scaled image.
    displayImage(qimg);

    // Update image area and reset object area
    int imageArea = inputImage.rows * inputImage.cols;
    ui.spinBoxImageArea->setValue(imageArea);
    ui.spinBoxObjectArea->setValue(0);
}

void ImageSegmentation::on_actionSave_triggered()
{
    if (outputObjectImage.empty()) {
        QMessageBox::warning(this, "Warning", "No processed image to save");
        return;
    }

    QString filename = QFileDialog::getSaveFileName(this, "Save Image", "",
        "PNG Image (*.png);;JPEG Image (*.jpg)");
    if (filename.isEmpty())
        return;

    const cv::Mat& saveMat = ui.toolButtonEdge->isChecked() ? outputEdgeImage : outputObjectImage;
    cv::imwrite(filename.toStdString(), saveMat);
}

void ImageSegmentation::on_actionProcessFolder_triggered() {
    QString inputDir = QFileDialog::getExistingDirectory(this, "Select Input Folder");
    if (inputDir.isEmpty()) return;

    QDir dir(inputDir);
    QString folderName = dir.dirName();
    QString parentDir = QFileInfo(inputDir).path();

    QString objectsDir = QString("%1/%2_objects").arg(parentDir).arg(folderName);
    QString redDir = QString("%1/%2_red").arg(parentDir).arg(folderName);
    QString edgesDir = QString("%1/%2_edges").arg(parentDir).arg(folderName);

    QDir().mkpath(objectsDir);
    QDir().mkpath(redDir);
    QDir().mkpath(edgesDir);

    QStringList filters = { "*.png", "*.jpg", "*.tif" };
    QFileInfoList files = QDir(inputDir).entryInfoList(filters, QDir::Files);

    foreach(const QFileInfo & fileInfo, files) {
        cv::Mat img = cv::imread(fileInfo.filePath().toStdString(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        SegmentationMode mode = currentMode;
        if (mode == SegmentationMode::Auto) {
            double minVal, maxVal;
            cv::minMaxLoc(img, &minVal, &maxVal);
            int count_light = 0, count_dark = 0;
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    uchar intensity = img.at<uchar>(i, j);
                    if (intensity == maxVal) count_light++;
                    if (intensity == minVal) count_dark++;
                }
            }
            mode = (count_light < count_dark) ? SegmentationMode::Light : SegmentationMode::Dark;
        }

        cv::Mat outputObject, outputEdges, objectMask;
        bool isLightObject;
        if (!segmentImage(img, mode, lambda, outputObject, outputEdges, objectMask, isLightObject)) continue;

        // Object on black/white background
        cv::Mat objectBackground;
        if (isLightObject) {
            objectBackground = cv::Mat::zeros(img.size(), CV_8UC1);
        }
        else {
            objectBackground = cv::Mat::ones(img.size(), CV_8UC1) * 255;
        }
        img.copyTo(objectBackground, objectMask);

        // Red object on white background
        cv::Mat redOnWhite = cv::Mat(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat redMask;
        cv::inRange(outputObject, cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 255), redMask);
        outputObject.copyTo(redOnWhite, redMask);

        // Edges on white background
        cv::Mat edgesOnWhite = cv::Mat(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat edgeMask;
        cv::inRange(outputEdges, cv::Scalar(0, 0, 250), cv::Scalar(0, 0, 255), edgeMask);
        outputEdges.copyTo(edgesOnWhite, edgeMask);

        QString baseName = fileInfo.baseName();
        cv::imwrite(QDir(objectsDir).filePath(baseName + ".png").toStdString(), objectBackground);
        cv::imwrite(QDir(redDir).filePath(baseName + ".png").toStdString(), redOnWhite);
        cv::imwrite(QDir(edgesDir).filePath(baseName + ".png").toStdString(), edgesOnWhite);
    }

    QMessageBox::information(this, "Complete", "Folder processing completed.");
}

void ImageSegmentation::on_pushButtonProcess_clicked()
{
    if (inputImage.empty()) {
        QMessageBox::warning(this, "Warning", "Please open an image first");
        return;
    }

    runSegmentation();
    updateDisplay();
}

void ImageSegmentation::runSegmentation()
{
    // ---------------------------
    // Build the graph from image.
    // ---------------------------
    GraphData data = createGraphData(inputImage, lambda, currentMode);
    int maxNodeId = 2 + data.m * data.n; // IDs: 0 (source), 1 (sink), then pixels.

    // ---------------------------
    // Compute max flow using Dinic's algorithm.
    // ---------------------------
    double flow = maxFlowDinic(data.graph, maxNodeId);
    qDebug() << "Max flow:" << flow;

    // --------------------------------------
    // Extract the segmentation (min cut).
    // --------------------------------------
    std::vector<bool> visited(maxNodeId, false);
    getSegmentationCut(data.graph, visited, maxNodeId);

    // --------------------------------------
    // Create output images based on cut.
    // --------------------------------------
    createOutputImages(inputImage, data.pixelNodes, visited, outputObjectImage, outputEdgeImage);

    // Calculate object area
    int objectArea = 0;
    for (int i = 0; i < data.m; ++i) {
        for (int j = 0; j < data.n; ++j) {
            ImageSegmentation::Node* node = data.pixelNodes[i][j];
            if (visited[node->id]) {
                objectArea++;
            }
        }
    }
    ui.spinBoxObjectArea->setValue(objectArea);

    // Clean up graph.
    delete data.graph;
}

void ImageSegmentation::updateDisplay()
{
    if (inputImage.empty())
        return;

    // Determine which image to display based on the state of the tool buttons
    if (!ui.toolButtonObject->isChecked() && !ui.toolButtonEdge->isChecked()) {
        // Show the original image
        QImage qimg = cvMatToQImage(inputImage);
        displayImage(qimg);
    }
    else if (ui.toolButtonObject->isChecked()) {
        // Show the segmented object image
        QImage qimg = cvMatToQImage(outputObjectImage);
        displayImage(qimg);
    }
    else if (ui.toolButtonEdge->isChecked()) {
        // Show the edge image
        QImage qimg = cvMatToQImage(outputEdgeImage);
        displayImage(qimg);
    }
}

QImage ImageSegmentation::cvMatToQImage(const cv::Mat& mat)
{
    // For CV_8UC1 (grayscale) images.
    if (mat.type() == CV_8UC1) {
        QImage image(mat.cols, mat.rows, QImage::Format_Grayscale8);
        for (int y = 0; y < mat.rows; ++y) {
            memcpy(image.scanLine(y), mat.ptr(y), static_cast<size_t>(mat.cols));
        }
        return image.copy();
    }
    // For CV_8UC3 (BGR) images.
    else if (mat.type() == CV_8UC3) {
        QImage image(mat.data, mat.cols, mat.rows,
            static_cast<int>(mat.step), QImage::Format_RGB888);
        return image.rgbSwapped().copy();
    }
    return QImage();
}

void ImageSegmentation::displayImage(const QImage& img)
{
    // Scale the image according to the current scaleFactor.
    QImage scaledImg = img.scaled(img.size() * scaleFactor,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation);
    ui.imageLabel->setPixmap(QPixmap::fromImage(scaledImg));
}

void ImageSegmentation::on_toolButtonObject_toggled(bool checked)
{
    if (checked) {
        // Uncheck the edge button
        ui.toolButtonEdge->setChecked(false);
    }
    // Update the display
    updateDisplay();
}

void ImageSegmentation::on_toolButtonEdge_toggled(bool checked)
{
    if (checked) {
        // Uncheck the object button
        ui.toolButtonObject->setChecked(false);
    }
    // Update the display
    updateDisplay();
}
