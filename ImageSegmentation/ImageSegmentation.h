#pragma once

#include <QtWidgets/QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QButtonGroup>
#include <opencv2/opencv.hpp>
#include "ui_ImageSegmentation.h"

class ImageSegmentation : public QMainWindow
{
    Q_OBJECT

public:
    ImageSegmentation(QWidget* parent = nullptr);
    ~ImageSegmentation();

    // Nested graph classes
    class Node;
    class Link;
    class Graph;

    enum class SegmentationMode { Light, Dark, Auto };

    bool segmentImage(const cv::Mat& input, SegmentationMode mode, double lambda,
        cv::Mat& outputObject, cv::Mat& outputEdges, cv::Mat& objectMask, bool& isLightObject);

private slots:
    void on_actionOpen_triggered();
    void on_actionSave_triggered();
    void on_actionProcessFolder_triggered();
    void on_pushButtonProcess_clicked();
    void updateDisplay();
    void on_toolButtonObject_toggled(bool checked);
    void on_toolButtonEdge_toggled(bool checked);

private:
    Ui::ImageSegmentationClass ui;

    // Image data
    cv::Mat inputImage;
    cv::Mat outputObjectImage;
    cv::Mat outputEdgeImage;
    QImage currentDisplay;

    // Mode handling
    SegmentationMode currentMode = SegmentationMode::Light;
    QButtonGroup modeButtonGroup;

    // Algorithm parameters
    double scaleFactor = 0.6;
    double lambda = 1;

    void runSegmentation();
    void displayImage(const QImage& img);
    QImage cvMatToQImage(const cv::Mat& mat);
};