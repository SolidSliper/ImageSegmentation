#include "ImageSegmentation.h"
#include "ImageIO.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QInputDialog>
#include <QElapsedTimer>
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QDir>
#include <QDirIterator>
#include <QProgressDialog>
#include <QFutureWatcher>
#include <QtConcurrent>

// ============================================================================
// Konstruktor / destruktor
// ============================================================================
ImageSegmentation::ImageSegmentation(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    // Tlacitka rezimu - mapovanie na enum
    modeButtonGroup.addButton(ui.toolButtonLight, static_cast<int>(Segmentation::Mode::Light));
    modeButtonGroup.addButton(ui.toolButtonDark,  static_cast<int>(Segmentation::Mode::Dark));
    modeButtonGroup.addButton(ui.toolButtonAuto,  static_cast<int>(Segmentation::Mode::Auto));

    connect(&modeButtonGroup, QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
            [this](QAbstractButton* button) {
                currentMode = static_cast<Segmentation::Mode>(modeButtonGroup.id(button));
                updateSeedIntensities();
            });

    connect(ui.toolButtonSelectROICustom, &QToolButton::toggled,
            this, &ImageSegmentation::on_toolButtonSelectROICustom_toggled);

    ui.toolButtonLight->setChecked(true);

    // Spinboxy pre parametre
    connect(ui.doubleSpinBoxLambda,    QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [this](double val) { lambda = val; });
    connect(ui.doubleSpinBoxScale,     QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [this](double val) { scaleFactor = val; });
    connect(ui.doubleSpinBoxPixelSize, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [this](double val) { pixelSizeNm = val; });
    connect(ui.spinBoxThreshold,       QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int val) { threshold = val; });

    ui.imageLabel->installEventFilter(this);
}

ImageSegmentation::~ImageSegmentation() = default;

// ============================================================================
// Pomocne UI metody
// ============================================================================

void ImageSegmentation::clearImageData()
{
    inputImage.release();
    outputObjectImage.release();
    outputEdgeImage.release();
    outputFeretImage.release();
    outputEllipseImage.release();
    outputMBRImage.release();
    lastObjectMask.release();
    userROIMask.release();
}

void ImageSegmentation::clearAllData()
{
    inputImage.release();
    outputObjectImage.release();
    outputEdgeImage.release();
    outputFeretImage.release();
    outputEllipseImage.release();
    outputMBRImage.release();
    lastObjectMask.release();
    userROIMask.release();

    segmentedObjectArea = 0;
    imageArea           = 0;
    userObject          = 0;
    userBackground      = 0;
    lastIsLightObject   = true;

    roiPolygonPoints.clear();
    polygonSelectionActive = false;
    roiSelectionActive     = false;
    roiFirstPointSet       = false;

    loadedImageName.clear();
    currentDisplay = QImage();

    ui.toolButtonSelectROIRectangle->setChecked(false);
    ui.toolButtonSelectROICustom->setChecked(false);
    ui.toolButtonLight->setChecked(true);

    ui.spinBoxImageArea->setValue(0);
    ui.spinBoxObjectArea->setValue(0);
    ui.doubleSpinBoxObject->setValue(0);
    ui.doubleSpinBoxBackground->setValue(0);

    ui.imageLabel->clear();
}

void ImageSegmentation::displayImage(const QImage& img)
{
    if (img.isNull()) return;
    QImage scaledImg = img.scaled(img.size() * scaleFactor,
                                  Qt::KeepAspectRatio,
                                  Qt::SmoothTransformation);
    ui.imageLabel->setPixmap(QPixmap::fromImage(scaledImg));
}

cv::Mat ImageSegmentation::applyROIMask(const cv::Mat& input)
{
    if (!userROIMask.empty())
        return userROIMask;
    return ImageIO::removeInfoOverlay(input);
}

Segmentation::Params ImageSegmentation::buildParams() const
{
    Segmentation::Params p;
    p.mode             = currentMode;
    p.algorithm        = (ui.comboBoxAlgorithm->currentText() == "Dinic")
                             ? Segmentation::Algorithm::Dinic
                             : Segmentation::Algorithm::EdmondsKarp;
    p.lambda           = lambda;
    p.userObject       = userObject;
    p.userBackground   = userBackground;
    p.noiseThreshold   = threshold;
    p.holeSize         = ui.spinBoxHoleSize->value();
    return p;
}

void ImageSegmentation::updateSeedIntensities()
{
    if (inputImage.empty()) return;

    const cv::Mat& mask = userROIMask.empty() ? cv::Mat() : userROIMask;
    Segmentation::computeSeedIntensities(inputImage, currentMode,
                                         defaultObjectIntensity,
                                         defaultBackgroundIntensity,
                                         mask);
    ui.doubleSpinBoxObject->setValue(defaultObjectIntensity);
    ui.doubleSpinBoxBackground->setValue(defaultBackgroundIntensity);
}

void ImageSegmentation::runSegmentation()
{
    userObject     = ui.doubleSpinBoxObject->value();
    userBackground = ui.doubleSpinBoxBackground->value();

    QElapsedTimer segTimer;
    segTimer.start();

    Segmentation::Result result;
    if (!Segmentation::segmentImage(inputImage, buildParams(), result, userROIMask)) {
        QMessageBox::warning(this, "Segmentation", "Segmentation failed");
        return;
    }

    qint64 elapsedMs = segTimer.elapsed();
    qDebug() << "Segmentation time:" << elapsedMs / 1000.0 << "s";
    ui.labelSegTime->setText(QString("Segmentation time: %1 s")
                                 .arg(elapsedMs / 1000.0, 0, 'f', 2));

    outputObjectImage   = result.objectImage;
    outputEdgeImage     = result.edgeImage;
    lastObjectMask      = result.objectMask;
    lastIsLightObject   = result.isLightObject;

    segmentedObjectArea = cv::countNonZero(lastObjectMask);
    ui.spinBoxObjectArea->setValue(segmentedObjectArea);
    imageArea = inputImage.rows * inputImage.cols;
    double perc = static_cast<double>(segmentedObjectArea) / imageArea * 100.0;
    ui.labelObjectArea->setText(QString("%1 %").arg(perc, 0, 'f', 2));

    // Geometricke parametre
    double longestFeret = 0.0, shortestFeret = 0.0, circleDiameter = 0.0;
    Segmentation::computeFeretDiameterAndCircle(inputImage, lastObjectMask,
                                                outputFeretImage,
                                                longestFeret, circleDiameter,
                                                shortestFeret);

    double majorAxis = 0.0, minorAxis = 0.0;
    Segmentation::computeLegendreEllipse(inputImage, lastObjectMask,
                                         outputEllipseImage,
                                         majorAxis, minorAxis);

    double longDiameter = 0.0, shortDiameter = 0.0;
    Segmentation::computeMBR(inputImage, lastObjectMask, outputMBRImage,
                             longDiameter, shortDiameter);

    on_actionEdge_triggered();
}

// ============================================================================
// Event filter - vyber ROI mysou
// ============================================================================
bool ImageSegmentation::eventFilter(QObject* obj, QEvent* event)
{
    if (obj != ui.imageLabel || event->type() != QEvent::MouseButtonPress)
        return QMainWindow::eventFilter(obj, event);

    QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
    QPoint pos = mouseEvent->pos();

    if (!ui.imageLabel->pixmap())
        return QMainWindow::eventFilter(obj, event);

    QPixmap pixmap = ui.imageLabel->pixmap();
    QSize pixSize   = pixmap.size();
    QSize labelSize = ui.imageLabel->size();
    int offsetX = (labelSize.width()  - pixSize.width())  / 2;
    int offsetY = (labelSize.height() - pixSize.height()) / 2;
    int relativeX = pos.x() - offsetX;
    int relativeY = pos.y() - offsetY;

    if (relativeX < 0 || relativeY < 0 ||
        relativeX >= pixSize.width() || relativeY >= pixSize.height())
        return true;

    double scale = static_cast<double>(pixSize.width()) / inputImage.cols;
    int origX = static_cast<int>(relativeX / scale);
    int origY = static_cast<int>(relativeY / scale);

    // Rectangle ROI
    if (roiSelectionActive) {
        if (!roiFirstPointSet) {
            roiFirstPoint    = cv::Point(origX, origY);
            roiFirstPointSet = true;
        } else {
            roiSecondPoint = cv::Point(origX, origY);
            int rx = std::min(roiFirstPoint.x, roiSecondPoint.x);
            int ry = std::min(roiFirstPoint.y, roiSecondPoint.y);
            int rw = std::abs(roiFirstPoint.x - roiSecondPoint.x);
            int rh = std::abs(roiFirstPoint.y - roiSecondPoint.y);
            cv::Rect roiRect(rx, ry, rw, rh);
            userROIMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
            userROIMask(roiRect).setTo(255);

            cv::Mat displayMat;
            if (inputImage.channels() == 1)
                cv::cvtColor(inputImage, displayMat, cv::COLOR_GRAY2BGR);
            else
                displayMat = inputImage.clone();
            cv::rectangle(displayMat, roiRect, cv::Scalar(144, 238, 144), 2);

            QImage qImage = ImageIO::cvMatToQImage(displayMat);
            currentDisplay = qImage;
            displayImage(qImage);

            /*outputFeretImage   = displayMat.clone();
            outputEllipseImage = displayMat.clone();
            outputMBRImage     = displayMat.clone();*/

            roiSelectionActive = false;
            roiFirstPointSet   = false;
            ui.toolButtonSelectROIRectangle->setChecked(false);
        }
        return true;
    }

    // Polygon ROI
    if (polygonSelectionActive) {
        if (mouseEvent->button() == Qt::LeftButton) {
            roiPolygonPoints.emplace_back(origX, origY);

            cv::Mat displayMat;
            if (inputImage.channels() == 1)
                cv::cvtColor(inputImage, displayMat, cv::COLOR_GRAY2BGR);
            else
                displayMat = inputImage.clone();
            for (size_t i = 0; i < roiPolygonPoints.size(); ++i) {
                cv::circle(displayMat, roiPolygonPoints[i], 3, cv::Scalar(0, 255, 0), -1);
                if (i > 0)
                    cv::line(displayMat, roiPolygonPoints[i - 1], roiPolygonPoints[i],
                             cv::Scalar(144, 238, 144), 2);
            }

            QImage qImage = ImageIO::cvMatToQImage(displayMat);
            currentDisplay = qImage;
            displayImage(qImage);

            /*outputFeretImage   = displayMat.clone();
            outputEllipseImage = displayMat.clone();
            outputMBRImage     = displayMat.clone();*/
        }
        else if (mouseEvent->button() == Qt::RightButton) {
            if (roiPolygonPoints.size() >= 3) {
                userROIMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
                std::vector<std::vector<cv::Point>> pts;
                pts.push_back(roiPolygonPoints);
                cv::fillPoly(userROIMask, pts, cv::Scalar(255));



                cv::Mat displayMat;
                if (inputImage.channels() == 1)
                    cv::cvtColor(inputImage, displayMat, cv::COLOR_GRAY2BGR);
                else
                    displayMat = inputImage.clone();
                const cv::Point* ptsArr[1] = { roiPolygonPoints.data() };
                int npts = static_cast<int>(roiPolygonPoints.size());
                cv::polylines(displayMat, ptsArr, &npts, 1, true,
                              cv::Scalar(144, 238, 144), 2);

                QImage qImage = ImageIO::cvMatToQImage(displayMat);
                currentDisplay = qImage;
                displayImage(qImage);

                //outputFeretImage   = displayMat.clone();
                //outputEllipseImage = displayMat.clone();
                //outputMBRImage     = displayMat.clone();

                polygonSelectionActive = false;
                ui.toolButtonSelectROICustom->setChecked(false);
            } else {
                roiPolygonPoints.clear();
                polygonSelectionActive = false;
                ui.toolButtonSelectROICustom->setChecked(false);
            }
        }
        return true;
    }
    return QMainWindow::eventFilter(obj, event);
}

// ============================================================================
// Subor: Open / Save
// ============================================================================
void ImageSegmentation::on_actionOpen_triggered()
{
    QString filename = QFileDialog::getOpenFileName(
        this, "Open Image",
        "D:/stu/bachelor/Kopani vyber Fe castica/",
        "Image Files (*.png *.jpg *.tif)");
    if (filename.isEmpty()) return;

    cv::Mat loadedImage = ImageIO::loadGrayscaleImage(filename);
    if (loadedImage.empty()) {
        QMessageBox::critical(this, "Error", "Failed to load image");
        return;
    }

    clearAllData();

    int ret = QMessageBox::question(this,
        "Remove an information overlay",
        "Does this image have an info overlay?",
        QMessageBox::Yes | QMessageBox::No);
    inputImage = (ret == QMessageBox::Yes)
                     ? ImageIO::removeInfoOverlay(loadedImage)
                     : loadedImage;

    QImage qimg = ImageIO::cvMatToQImage(inputImage);
    displayImage(qimg);

    imageArea = inputImage.rows * inputImage.cols;
    ui.spinBoxImageArea->setValue(imageArea);
    ui.spinBoxObjectArea->setValue(0);

    Segmentation::computeSeedIntensities(inputImage, currentMode,
                                         defaultObjectIntensity,
                                         defaultBackgroundIntensity);
    ui.doubleSpinBoxObject->setValue(defaultObjectIntensity);
    ui.doubleSpinBoxBackground->setValue(defaultBackgroundIntensity);

    // Default ROI on whole image    
    int imgWidth = inputImage.size().width;
    int imgHeight = inputImage.size().height;
    cv::Rect roiRect(0, 0, imgWidth - 1, imgHeight - 1);
    userROIMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
    userROIMask(roiRect).setTo(255);

    userROIMask.release();

    QFileInfo fi(filename);
    loadedImageName = fi.baseName();

    updateSeedIntensities();
}

void ImageSegmentation::on_actionSave_triggered()
{
    QString defaultName = loadedImageName.isEmpty()
                              ? "output_overlay"
                              : loadedImageName + "_overlay";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Image with Overlays",
        "D:/stu/bachelor/Kopani vyber Fe castica/" + defaultName + ".tif",
        "TIF Image (*.tif)");
    if (filename.isEmpty()) return;

    cv::Mat saveMat = ImageIO::QImageToCvMat(currentDisplay);
    ImageIO::saveImage(filename, saveMat);
}

void ImageSegmentation::on_actionSaveObject_triggered()
{
    if (inputImage.empty() || lastObjectMask.empty()) {
        QMessageBox::warning(this, "Warning", "No segmented object available to save");
        return;
    }

    cv::Mat segmentedImage = lastIsLightObject
        ? cv::Mat::zeros(inputImage.size(), inputImage.type())
        : cv::Mat::ones(inputImage.size(), inputImage.type()) * 255;
    inputImage.copyTo(segmentedImage, lastObjectMask);

    QString defaultName = loadedImageName.isEmpty()
                              ? "segmented_object"
                              : loadedImageName + "_object";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Segmented Object",
        defaultName + ".png",
        "PNG Image (*.png);;JPEG Image (*.jpg)");
    if (filename.isEmpty()) return;

    ImageIO::saveImage(filename, segmentedImage);
}

void ImageSegmentation::on_actionSaveAllStates_triggered()
{
    QString folder = QFileDialog::getExistingDirectory(this,
        "Select Folder to Save All States",
        QDir::homePath());
    if (folder.isEmpty()) return;

    QString baseName = loadedImageName.isEmpty() ? "output" : loadedImageName;
    bool ok;
    QString customPrefix = QInputDialog::getText(this,
        "Save All States",
        "Enter file name prefix:",
        QLineEdit::Normal, baseName, &ok);
    if (ok && !customPrefix.isEmpty())
        baseName = customPrefix;

    if (!inputImage.empty() && !lastObjectMask.empty()) {
        cv::Mat segmentedImage = lastIsLightObject
            ? cv::Mat::zeros(inputImage.size(), inputImage.type())
            : cv::Mat::ones(inputImage.size(), inputImage.type()) * 255;
        inputImage.copyTo(segmentedImage, lastObjectMask);
        ImageIO::saveImage(folder + "/" + baseName + "_object.png", segmentedImage);
    } else {
        QMessageBox::warning(this, "Save All States", "No segmented object available to save.");
    }

    if (!outputEdgeImage.empty())
        ImageIO::saveImage(folder + "/" + baseName + "_edge.png",    outputEdgeImage);
    if (!outputFeretImage.empty())
        ImageIO::saveImage(folder + "/" + baseName + "_feret.png",   outputFeretImage);
    if (!outputEllipseImage.empty())
        ImageIO::saveImage(folder + "/" + baseName + "_ellipse.png", outputEllipseImage);
    if (!outputMBRImage.empty())
        ImageIO::saveImage(folder + "/" + baseName + "_MBR.png",     outputMBRImage);

    QMessageBox::information(this, "Save All States",
        "All available state images have been saved to:\n" + folder);
}

void ImageSegmentation::on_actionProcessFolder_triggered()
{
    QString folder = QFileDialog::getExistingDirectory(
        this,
        "Select Folder for Processing",
        QDir::homePath());

    if (folder.isEmpty())
        return;

    QDir(folder).mkdir("processed");

    QStringList files;

    QDirIterator it(
        folder,
        QStringList() << "*.jpg" << "*.png",
        QDir::Files);

    while (it.hasNext())
        files << it.next();

    const int total = files.size();

    if (total == 0)
        return;

    cv::Mat savedROI = userROIMask.clone();

    auto* dialog = new QProgressDialog(
        "Remaining images: " + QString::number(total),
        nullptr,
        0,
        total,
        this);

    dialog->setWindowModality(Qt::WindowModal);
    dialog->setMinimumDuration(0);
    dialog->setAutoClose(true);


    auto* watcher = new QFutureWatcher<void>(this);

    QFuture<void> future = QtConcurrent::run(
        [this, files, dialog, total, folder, savedROI]()
        {
            int done = 0;

            for (const QString& file : files)
            {
                inputImage = ImageIO::loadGrayscaleImage(file);
                QFileInfo fi(file);
                loadedImageName = fi.baseName();
                QString baseName = loadedImageName.isEmpty() ? "output" : loadedImageName;

                if (!savedROI.empty() && savedROI.size() == inputImage.size())
                    userROIMask = savedROI.clone();

                runSegmentation();

                if (!inputImage.empty() && !lastObjectMask.empty()) {
                    cv::Mat segmentedImage = lastIsLightObject
                        ? cv::Mat::zeros(inputImage.size(), inputImage.type())
                        : cv::Mat::ones(inputImage.size(), inputImage.type()) * 255;
                    inputImage.copyTo(segmentedImage, lastObjectMask);
                    ImageIO::saveImage(folder + "/processed/" + baseName + "_object.png", segmentedImage);
                }
                else {
                    QMessageBox::warning(this, "Save All States", "No segmented object available to save.");
                }

                if (!outputEdgeImage.empty())
                    ImageIO::saveImage(folder + "/processed/" + baseName + "_edge.png", outputEdgeImage);
                if (!outputFeretImage.empty())
                    ImageIO::saveImage(folder + "/processed/" + baseName + "_feret.png", outputFeretImage);
                if (!outputEllipseImage.empty())
                    ImageIO::saveImage(folder + "/processed/" + baseName + "_ellipse.png", outputEllipseImage);
                if (!outputMBRImage.empty())
                    ImageIO::saveImage(folder + "/processed/" + baseName + "_MBR.png", outputMBRImage);

                done++;

                int remaining = total - done;

                QMetaObject::invokeMethod(
                    dialog,
                    [dialog, done, remaining]()
                    {
                        dialog->setValue(done);
                        dialog->setLabelText(
                            QString("Remaining images: %1")
                            .arg(remaining));
                    },
                    Qt::QueuedConnection);
            }
        });

    connect(
        watcher,
        &QFutureWatcher<void>::finished,
        dialog,
        &QProgressDialog::close);

    connect(
        watcher,
        &QFutureWatcher<void>::finished,
        watcher,
        &QObject::deleteLater);

    watcher->setFuture(future);

    dialog->show();
}

void ImageSegmentation::on_actionSaveInfo_triggered()
{
    if (inputImage.empty() || lastObjectMask.empty()) {
        QMessageBox::warning(this, "Warning", "No segmented object available to compute info");
        return;
    }

    int    area        = segmentedObjectArea;
    double areaPercent = (imageArea > 0) ? (area * 100.0 / imageArea) : 0.0;
    double meanGray    = cv::mean(inputImage, lastObjectMask)[0];

    cv::Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&inputImage, 1, 0, lastObjectMask, hist, 1, &histSize, &histRange);

    double maxHistVal = 0;
    int    modalGray  = 0;
    for (int i = 0; i < histSize; i++) {
        float hVal = hist.at<float>(i);
        if (hVal > maxHistVal) { maxHistVal = hVal; modalGray = i; }
    }

    double minGray, maxGray;
    cv::minMaxLoc(inputImage, &minGray, &maxGray, nullptr, nullptr, lastObjectMask);

    double sumX = 0, sumY = 0;
    int count = 0;
    for (int i = 0; i < lastObjectMask.rows; i++) {
        for (int j = 0; j < lastObjectMask.cols; j++) {
            if (lastObjectMask.at<uchar>(i, j) > 0) {
                sumX += j; sumY += i; count++;
            }
        }
    }
    double centroidX = (count > 0) ? (sumX / count) : 0;
    double centroidY = (count > 0) ? (sumY / count) : 0;

    int nedge = 0;
    for (int i = 0; i < lastObjectMask.rows; i++) {
        for (int j = 0; j < lastObjectMask.cols; j++) {
            if (lastObjectMask.at<uchar>(i, j) > 0) {
                bool isBoundary = false;
                /*if (i > 0 && lastObjectMask.at<uchar>(i - 1, j) == 0) isBoundary = true;
                if (i < lastObjectMask.rows - 1 && lastObjectMask.at<uchar>(i + 1, j) == 0) isBoundary = true;
                if (j > 0 && lastObjectMask.at<uchar>(i, j - 1) == 0) isBoundary = true;
                if (j < lastObjectMask.cols - 1 && lastObjectMask.at<uchar>(i, j + 1) == 0) isBoundary = true;
                if (isBoundary) nedge++;*/
                if (i > 0 && lastObjectMask.at<uchar>(i - 1, j) == 0) nedge++;
                if (i < lastObjectMask.rows - 1 && lastObjectMask.at<uchar>(i + 1, j) == 0) nedge++;
                if (j > 0 && lastObjectMask.at<uchar>(i, j - 1) == 0) nedge++;
                if (j < lastObjectMask.cols - 1 && lastObjectMask.at<uchar>(i, j + 1) == 0) nedge++;
            }
        }
    }
    double perimeter = static_cast<double>(nedge); 

    double longestFeret = 0.0, shortestFeret = 0.0, circleDiameter = 0.0;
    cv::Mat dummyAnnotated;
    Segmentation::computeFeretDiameterAndCircle(inputImage, lastObjectMask,
                                                dummyAnnotated, longestFeret,
                                                circleDiameter, shortestFeret);

    double majorAxis = 0.0, minorAxis = 0.0;
    Segmentation::computeLegendreEllipse(inputImage, lastObjectMask,
                                         dummyAnnotated, majorAxis, minorAxis);
    double ellipseRatio = (majorAxis != 0) ? (minorAxis / majorAxis) : 0.0;

    double longDiameter = 0.0, shortDiameter = 0.0;
    Segmentation::computeMBR(inputImage, lastObjectMask, dummyAnnotated,
                             longDiameter, shortDiameter);
    double LWRatio     = (longDiameter   != 0) ? (shortDiameter / longDiameter) : 0.0;
    double aspectRatio = (longestFeret   != 0) ? (shortestFeret / longestFeret) : 0.0;
    double perimEqDia  = (perimeter      > 0)  ? (perimeter / M_PI)             : 0.0;
    double circularity = (perimEqDia     > 0)  ? (circleDiameter / perimEqDia)  : 0.0;

    double pSize           = ui.doubleSpinBoxPixelSize->value();
    double area_nm2        = area          * pSize;
    double perimeter_nm    = perimeter     * pSize;
    double longestFeret_nm  = longestFeret  * pSize;
    double shortestFeret_nm = shortestFeret * pSize;
    double majorAxis_nm    = majorAxis     * pSize;
    double minorAxis_nm    = minorAxis     * pSize;
    double longDiameter_nm  = longDiameter  * pSize;
    double shortDiameter_nm = shortDiameter * pSize;

    QString info;
    if (ui.checkBoxArea->isChecked())
        info += "Area: " + QString::number(area) + " pixels (" + QString::number(area_nm2, 'f', 2) + " nm^2)\n";
    if (ui.checkBoxMeanGrayValue->isChecked())
        info += "Mean Gray Value: " + QString::number(meanGray, 'f', 2) + "\n";
    if (ui.checkBoxModalGrayValue->isChecked())
        info += "Modal Gray Value: " + QString::number(modalGray) + "\n";
    if (ui.checkBoxMinMax->isChecked())
        info += "Min Gray Level: " + QString::number(minGray) + ", Max Gray Level: " + QString::number(maxGray) + "\n";
    if (ui.checkBoxCentroid->isChecked())
        info += "Centroid: (" + QString::number(centroidX, 'f', 2) + ", " + QString::number(centroidY, 'f', 2) + ")\n";
    if (ui.checkBoxCircularity->isChecked())
        info += "Circularity: " + QString::number(circularity, 'f', 3) + "\n";
    if (ui.checkBoxPerimeter->isChecked())
        info += "Perimeter: " + QString::number(perimeter) + " pixels (" + QString::number(perimeter_nm, 'f', 2) + " nm)\n";
    if (ui.checkBoxFeret->isChecked())
        info += "Feret diameters: " + QString::number(longestFeret) + " / " + QString::number(shortestFeret) +
                " pixels (" + QString::number(longestFeret_nm, 'f', 2) + " / " + QString::number(shortestFeret_nm, 'f', 2) + " nm)\n";
    if (ui.checkBoxCircularity->isChecked())
        info += "Area-equivalent circle diameter: " + QString::number(circleDiameter) + " pixels\n";
    if (ui.checkBoxEllipseRatio->isChecked())
        info += "Ellipse Ratio (Minor/Major): " + QString::number(ellipseRatio, 'f', 2) + "\n";
    if (ui.checkBoxLWRatio->isChecked())
        info += "L/W Ratio of MBR: " + QString::number(LWRatio, 'f', 2) + "\n";
    if (ui.checkBoxAspectRatio->isChecked())
        info += "Aspect Ratio (Shortest/Longest Feret): " + QString::number(aspectRatio, 'f', 2) + "\n";

    QString defaultName = loadedImageName.isEmpty()
                              ? "selection_info"
                              : loadedImageName + "_info";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Selection Info", defaultName + ".txt", "Text Files (*.txt)");
    if (filename.isEmpty()) return;

    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << info;
        file.close();
        QMessageBox::information(this, "Save Info", "Information saved successfully.");
    }
}

// ============================================================================
// Spracovanie a prepinanie zobrazeni
// ============================================================================

void ImageSegmentation::on_pushButtonProcess_clicked()
{
    if (inputImage.empty()) {
        QMessageBox::warning(this, "Warning", "Please open an image first");
        return;
    }
    runSegmentation();
}

void ImageSegmentation::on_pushButtonScale_clicked()
{
    scaleFactor = ui.doubleSpinBoxScale->value();
    displayImage(currentDisplay);
}

void ImageSegmentation::on_actionOriginal_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(inputImage);
    displayImage(qImage);
}

void ImageSegmentation::on_actionFeret_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(outputFeretImage);
    currentDisplay = qImage;
    displayImage(qImage);
}

void ImageSegmentation::on_actionEdge_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(outputEdgeImage);
    currentDisplay = qImage;
    displayImage(qImage);
}

void ImageSegmentation::on_actionObject_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(outputObjectImage);
    currentDisplay = qImage;
    displayImage(qImage);
}

void ImageSegmentation::on_actionEllipse_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(outputEllipseImage);
    currentDisplay = qImage;
    displayImage(qImage);
}

void ImageSegmentation::on_actionMBR_triggered()
{
    QImage qImage = ImageIO::cvMatToQImage(outputMBRImage);
    currentDisplay = qImage;
    displayImage(qImage);
}

// ============================================================================
// Slider <-> SpinBox synchronizacia
// ============================================================================

void ImageSegmentation::on_horizontalSliderBackground_valueChanged()
{
    ui.doubleSpinBoxBackground->setValue(ui.horizontalSliderBackground->value());
}

void ImageSegmentation::on_horizontalSliderObject_valueChanged()
{
    ui.doubleSpinBoxObject->setValue(ui.horizontalSliderObject->value());
}

void ImageSegmentation::on_doubleSpinBoxBackground_valueChanged()
{
    ui.horizontalSliderBackground->setValue(ui.doubleSpinBoxBackground->value());
}

void ImageSegmentation::on_doubleSpinBoxObject_valueChanged()
{
    ui.horizontalSliderObject->setValue(ui.doubleSpinBoxObject->value());
}

// ============================================================================
// ROI tlacitka
// ============================================================================

void ImageSegmentation::on_toolButtonSelectROIRectangle_toggled(bool checked)
{
    if (checked) {
        roiSelectionActive = true;
        roiFirstPointSet   = false;
    } else {
        roiSelectionActive = false;
        roiFirstPointSet   = false;
        updateSeedIntensities();
    }
}

void ImageSegmentation::on_toolButtonSelectROICustom_toggled(bool checked)
{
    if (checked) {
        polygonSelectionActive = true;
        roiPolygonPoints.clear();
        roiSelectionActive = false;
        ui.toolButtonSelectROIRectangle->setChecked(false);
    } else {
        polygonSelectionActive = false;
        if (!roiPolygonPoints.empty() && roiPolygonPoints.size() < 3)
            roiPolygonPoints.clear();
        updateSeedIntensities();
    }
}
