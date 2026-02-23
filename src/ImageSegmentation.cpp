#include "ImageSegmentation.h"
//#include "ImageSegmentationMath.cpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QDir>
#include <QFileInfo>
#include <QCoreApplication>
#include <QMouseEvent>
#include <QFile>
#include <QTextStream>
#include <QInputDialog>

// Konstruktor, destruktor a inicializacia UI
ImageSegmentation::ImageSegmentation(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    // Pridanie tlacidiel pre rezimy Light, Dark, Auto do skupiny
    modeButtonGroup.addButton(ui.toolButtonLight, static_cast<int>(SegmentationMode::Light));
    modeButtonGroup.addButton(ui.toolButtonDark, static_cast<int>(SegmentationMode::Dark));
    modeButtonGroup.addButton(ui.toolButtonAuto, static_cast<int>(SegmentationMode::Auto));
    // Pri kliknuti na tlacidlo prepni rezim a prepocitaj seed intenzity
    connect(&modeButtonGroup, QOverload<QAbstractButton*>::of(&QButtonGroup::buttonClicked),
        [this](QAbstractButton* button)
        {
            currentMode = static_cast<SegmentationMode>(modeButtonGroup.id(button));
            updateSeedIntensities();
        });

    connect(ui.toolButtonSelectROICustom, &QToolButton::toggled,
        this, &ImageSegmentation::on_toolButtonSelectROICustom_toggled);

    ui.toolButtonLight->setChecked(true); // predvoleny rezim
    // Prepojenie spinboxov na zmenu parametrov
    connect(ui.doubleSpinBoxLambda, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double val) { lambda = val; });
    connect(ui.doubleSpinBoxScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double val) { scaleFactor = val; });
    connect(ui.doubleSpinBoxPixelSize, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        [this](double val) { pixelSizeNm = val; });
    connect(ui.spinBoxThreshold, QOverload<int>::of(&QSpinBox::valueChanged),
        [this](double val) { threshold = val; });
    // Nainstalovanie event filteru na label pre ROI vyber
    ui.imageLabel->installEventFilter(this);
}

// Destruktor
ImageSegmentation::~ImageSegmentation() {}

// Vycisti vsetky interni data pri otvoreni noveho obrazka
void ImageSegmentation::clearAllData()
{
    // Release image mats
    inputImage.release();
    outputObjectImage.release();
    outputEdgeImage.release();
    outputFeretImage.release();
    outputEllipseImage.release();
    outputMBRImage.release();
    lastObjectMask.release();
    userROIMask.release();

    // Reset numeric/state variables
    segmentedObjectArea = 0;
    imageArea = 0;
    userObject = 0;
    userBackground = 0;
    lastIsLightObject = true;
    roiPolygonPoints.clear();
    polygonSelectionActive = false;
    roiSelectionActive = false;
    roiFirstPointSet = false;

    // Reset UI related state
    loadedImageName.clear();
    currentDisplay = QImage();

    // Reset tool buttons and UI fields
    ui.toolButtonSelectROIRectangle->setChecked(false);
    ui.toolButtonSelectROICustom->setChecked(false);
    ui.toolButtonObject->setChecked(false);
    ui.toolButtonEdge->setChecked(false);
    ui.toolButtonFeret->setChecked(false);
    ui.toolButtonEllipse->setChecked(false);
    ui.toolButtonMBR->setChecked(false);
    ui.toolButtonLight->setChecked(true);

    ui.spinBoxImageArea->setValue(0);
    ui.spinBoxObjectArea->setValue(0);
    ui.doubleSpinBoxObject->setValue(0);
    ui.doubleSpinBoxBackground->setValue(0);

    // Clear display
    ui.imageLabel->clear();
}

// Prevod cv::Mat na QImage, podporuje jednokanálovy aj trojkanálovy mat
QImage ImageSegmentation::cvMatToQImage(const cv::Mat& mat)
{
    if (mat.empty())
        return QImage();

    switch (mat.type()) {
    case CV_8UC1: {
        // sivy obrazok
        QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return img.copy();
    }
    case CV_8UC3: {
        // farebny obrazok BGR -> RGB
        cv::Mat tmp;
        cv::cvtColor(mat, tmp, cv::COLOR_BGR2RGB);
        QImage img(tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_RGB888);
        return img.copy();
    }
    default:
        // preved na sivy
        cv::Mat grey;
        mat.convertTo(grey, CV_8U);
        QImage img(grey.data, grey.cols, grey.rows, grey.step, QImage::Format_Grayscale8);
        return img.copy();
    }
}

// Zobrazenie obrazka v labeli, aplikacia mierky a zachovanie pomeru stran
void ImageSegmentation::displayImage(const QImage& img)
{
    if (img.isNull()) return;
    QImage scaledImg = img.scaled(img.size() * scaleFactor,
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation);
    ui.imageLabel->setPixmap(QPixmap::fromImage(scaledImg));
}

// Prepne zobrazenie podla vybraneho tlacidla
void ImageSegmentation::updateDisplay()
{
    QImage qImage;
    if (ui.toolButtonFeret->isChecked() && !outputFeretImage.empty())
        qImage = cvMatToQImage(outputFeretImage);
    else if (ui.toolButtonEllipse->isChecked() && !outputEllipseImage.empty())
        qImage = cvMatToQImage(outputEllipseImage);
    else if (ui.toolButtonMBR->isChecked() && !outputMBRImage.empty())
        qImage = cvMatToQImage(outputMBRImage);
    else if (ui.toolButtonObject->isChecked())
        qImage = cvMatToQImage(outputObjectImage);
    else if (ui.toolButtonEdge->isChecked())
        qImage = cvMatToQImage(outputEdgeImage);
    else
        qImage = cvMatToQImage(inputImage);
    displayImage(qImage);
}

// Zrusenie stavu ostatnych tlacidiel pri vybere jedneho
void ImageSegmentation::resetToolButtons(QAbstractButton* active)
{
    QList<QAbstractButton*> buttons = { ui.toolButtonObject,
                                        ui.toolButtonEdge,
                                        ui.toolButtonFeret,
                                        ui.toolButtonEllipse,
                                        ui.toolButtonMBR };
    for (QAbstractButton* button : buttons)
        if (button != active)
            button->setChecked(false);
}

// Vratenie ROI masky: pouzije uzivatelovu, ak existuje; inak odstranenie overlay
cv::Mat ImageSegmentation::applyROIMask(const cv::Mat& input)
{
    if (!userROIMask.empty())
        return userROIMask;
    return removeInfoOverlay(input);
}

// Odstrani spodnych 10% obrazka ako informacny overlay
cv::Mat ImageSegmentation::removeInfoOverlay(const cv::Mat& input)
{
    int overlayHeight = input.rows / 9;
    cv::Rect cropRect(0, 0, input.cols, input.rows - overlayHeight);
    return input(cropRect).clone();
}

// Hlavny wrapper pre segmentaciu, spracuje vystupy a vykresli statistiky
void ImageSegmentation::runSegmentation()
{
    // ziskanie hodnot od uzivatela
    userObject = ui.doubleSpinBoxObject->value();
    userBackground = ui.doubleSpinBoxBackground->value();

    cv::Mat outObj, outEdge, mask;
    bool isLight;
    cv::Mat roiApplied = applyROIMask(inputImage);

    // volanie segmentacnej funkcie
    if (!segmentImage(inputImage, currentMode, lambda,
        outObj, outEdge, mask, isLight, roiApplied)) {
        QMessageBox::warning(this, "Segmentation", "Segmentation failed");
        return;
    }

    //// najdenie najvacsieho konturu na dalsiu analyzu
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //if (!contours.empty()) {
    //    int maxIdx = 0;
    //    double maxA = 0;
    //    for (int i = 0; i < static_cast<int>(contours.size()); i++) {
    //        double a = cv::contourArea(contours[i]);
    //        if (a > maxA) { maxA = a; maxIdx = i; }
    //    }
    //}

    // ulozenie vystupnych obrazkov a statistiky
    outputObjectImage = outObj;
    outputEdgeImage = outEdge;
    segmentedObjectArea = cv::countNonZero(mask);
    ui.spinBoxObjectArea->setValue(segmentedObjectArea);
    imageArea = inputImage.rows * inputImage.cols;
    double perc = static_cast<double>(segmentedObjectArea) / imageArea * 100.0;
    ui.labelObjectArea->setText(QString("%1 %").arg(perc, 0, 'f', 2));

    // vypocet geometrickych parametrov
    {
        double longestFeret = 0.0, shortestFeret = 0.0, circleDiameter = 0.0;
        computeFeretDiameterAndCircle(inputImage, mask,
            outputFeretImage,
            longestFeret,
            circleDiameter,
            shortestFeret);
        /*ui.labelFeretResults->setText(QString("Feret diameters: %1 / %2, Circle diameter: %3")
            .arg(longestFeret, 0, 'f', 2)
            .arg(shortestFeret, 0, 'f', 2)
            .arg(circleDiameter, 0, 'f', 2));*/
    }
    {
        double majorAxis = 0.0, minorAxis = 0.0;
        computeLegendreEllipse(inputImage, mask, outputEllipseImage, majorAxis, minorAxis);
    }
    {
        double longDiameter = 0.0, shortDiameter = 0.0;
        computeMBR(inputImage, mask, outputMBRImage, longDiameter, shortDiameter);
    }

    lastObjectMask = mask;
    lastIsLightObject = isLight;
}

 //Interaktivny vyber ROI pomocou mysich udalosti
bool ImageSegmentation::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == ui.imageLabel) {
        if (event->type() == QEvent::MouseButtonPress) {
            QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
            QPoint pos = mouseEvent->pos();
            // ak nie je pixmapa, nic nechytaj
            if (!ui.imageLabel->pixmap())
                return QMainWindow::eventFilter(obj, event);
            QPixmap pixmap = ui.imageLabel->pixmap();
            QSize pixSize = pixmap.size();
            QSize labelSize = ui.imageLabel->size();
            // vypocet offsetu pre centrovanie
            int offsetX = (labelSize.width() - pixSize.width()) / 2;
            int offsetY = (labelSize.height() - pixSize.height()) / 2;
            int relativeX = pos.x() - offsetX;
            int relativeY = pos.y() - offsetY;
            // mimo obrazka ignoruj
            if (relativeX < 0 || relativeY < 0 || relativeX >= pixSize.width() || relativeY >= pixSize.height())
                return true;
            // prevod na povodne suradnice
            double scale = static_cast<double>(pixSize.width()) / inputImage.cols;
            int origX = static_cast<int>(relativeX / scale);
            int origY = static_cast<int>(relativeY / scale);

            // Rectangle ROI selection
            if (roiSelectionActive) {
                if (!roiFirstPointSet) {
                    // nastav prvy bod
                    roiFirstPoint = cv::Point(origX, origY);
                    roiFirstPointSet = true;
                }
                else {
                    // nastav druhy bod a vytvor masku
                    roiSecondPoint = cv::Point(origX, origY);
                    int rx = std::min(roiFirstPoint.x, roiSecondPoint.x);
                    int ry = std::min(roiFirstPoint.y, roiSecondPoint.y);
                    int rw = std::abs(roiFirstPoint.x - roiSecondPoint.x);
                    int rh = std::abs(roiFirstPoint.y - roiSecondPoint.y);
                    cv::Rect roiRect(rx, ry, rw, rh);
                    userROIMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
                    userROIMask(roiRect).setTo(255);
                    // vykresli ramec pre vizualizaciu ROI
                    cv::Mat displayImage;
                    if (inputImage.channels() == 1)
                        cv::cvtColor(inputImage, displayImage, cv::COLOR_GRAY2BGR);
                    else
                        displayImage = inputImage.clone();
                    cv::rectangle(displayImage, roiRect, cv::Scalar(144, 238, 144), 2);
                    outputFeretImage = displayImage.clone();
                    outputEllipseImage = displayImage.clone();
                    outputMBRImage = displayImage.clone();
                    // ukonci rezim vyberu
                    roiSelectionActive = false;
                    roiFirstPointSet = false;
                    ui.toolButtonSelectROIRectangle->setChecked(false);
                    updateDisplay();
                }
                return true;
            }

            // Polygon (custom) ROI selection
            if (polygonSelectionActive) {
                // left click adds a point
                if (mouseEvent->button() == Qt::LeftButton) {
                    roiPolygonPoints.emplace_back(origX, origY);
                    // draw temporary polygon/points for visualization
                    cv::Mat displayImage;
                    if (inputImage.channels() == 1)
                        cv::cvtColor(inputImage, displayImage, cv::COLOR_GRAY2BGR);
                    else
                        displayImage = inputImage.clone();
                    // draw existing polygon edges and points
                    for (size_t i = 0; i < roiPolygonPoints.size(); ++i) {
                        cv::circle(displayImage, roiPolygonPoints[i], 3, cv::Scalar(0, 255, 0), -1);
                        if (i > 0)
                            cv::line(displayImage, roiPolygonPoints[i-1], roiPolygonPoints[i], cv::Scalar(144, 238, 144), 2);
                    }
                    outputFeretImage = displayImage.clone();
                    outputEllipseImage = displayImage.clone();
                    outputMBRImage = displayImage.clone();
                }
                // right click finishes polygon
                else if (mouseEvent->button() == Qt::RightButton) {
                    if (roiPolygonPoints.size() >= 3) {
                        userROIMask = cv::Mat::zeros(inputImage.size(), CV_8UC1);
                        std::vector<std::vector<cv::Point>> pts;
                        pts.push_back(roiPolygonPoints);
                        cv::fillPoly(userROIMask, pts, cv::Scalar(255));
                        // draw final polygon on display
                        cv::Mat displayImage;
                        if (inputImage.channels() == 1)
                            cv::cvtColor(inputImage, displayImage, cv::COLOR_GRAY2BGR);
                        else
                            displayImage = inputImage.clone();
                        const cv::Point* ptsArr[1] = { roiPolygonPoints.data() };
                        int npts = static_cast<int>(roiPolygonPoints.size());
                        cv::polylines(displayImage, ptsArr, &npts, 1, true, cv::Scalar(144, 238, 144), 2);
                        outputFeretImage = displayImage.clone();
                        outputEllipseImage = displayImage.clone();
                        outputMBRImage = displayImage.clone();
                        // finish
                        polygonSelectionActive = false;
                        ui.toolButtonSelectROICustom->setChecked(false);
                        updateDisplay();
                    }
                    else {
                        // not enough points, clear
                        roiPolygonPoints.clear();
                        polygonSelectionActive = false;
                        ui.toolButtonSelectROICustom->setChecked(false);
                    }
                }
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

// Akcie pre menu Subor: otvorenie, ulozenie, atd.
void ImageSegmentation::on_actionOpen_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open Image", "D:/article/military-data-2026",
        "Image Files (*.png *.jpg *.tif)");
    if (filename.isEmpty())
        return;
    cv::Mat loadedImage = cv::imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
    if (loadedImage.empty()) {
        QMessageBox::critical(this, "Error", "Failed to load image");
        return;
    }

    // Clear previous state before applying the newly loaded image
    clearAllData();

    // opytaj sa na odstranenie info overlay
    bool removeOverlay = false;
    int ret = QMessageBox::question(this,
        "Remove an information overlay",
        "Does this image have an info overlay?",
        QMessageBox::Yes | QMessageBox::No);
    if (ret == QMessageBox::Yes)
        removeOverlay = true;

    // aplikuj overlay removal ak ziadane
    if (removeOverlay)
        inputImage = removeInfoOverlay(loadedImage);
    else
        inputImage = loadedImage;

    // zobraz vstupny obraz
    QImage qimg = cvMatToQImage(inputImage);
    displayImage(qimg);

    // nastav plochu obrazu v UI
    imageArea = inputImage.rows * inputImage.cols;
    ui.spinBoxImageArea->setValue(imageArea);
    ui.spinBoxObjectArea->setValue(0);

    // predvolene seed intenzity
    computeSeedIntensities(inputImage, currentMode, defaultObjectIntensity, defaultBackgroundIntensity);
    ui.doubleSpinBoxObject->setValue(defaultObjectIntensity);
    ui.doubleSpinBoxBackground->setValue(defaultBackgroundIntensity);
    // zrus existujucu ROI masku
    userROIMask.release();

    QFileInfo fi(filename);
    loadedImageName = fi.baseName();

    // aktualizuj seed intenzity
    updateSeedIntensities();
}

// Ulozenie obrazka s prekryvmi do suboru
void ImageSegmentation::on_actionSave_triggered()
{
    // navrh nazvu suboru podla mena obrazka
    QString defaultName = loadedImageName.isEmpty() ? "output_overlay" : loadedImageName + "_overlay";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Image with Overlays",
        defaultName + ".tif",
        "TIF Image (*.tif)");
    if (filename.isEmpty())
        return;

    // vyber, ktory obrazok sa ulozi podla aktivneho tlacidla
    cv::Mat saveMat;
    if ((ui.toolButtonFeret->isChecked() && !outputFeretImage.empty()) ||
        (ui.toolButtonEllipse->isChecked() && !outputEllipseImage.empty()) ||
        (ui.toolButtonMBR->isChecked() && !outputMBRImage.empty()))
    {
        if (ui.toolButtonFeret->isChecked())
            saveMat = outputFeretImage;
        else if (ui.toolButtonEllipse->isChecked())
            saveMat = outputEllipseImage;
        else
            saveMat = outputMBRImage;
    }
    else if (ui.toolButtonObject->isChecked())
        saveMat = outputObjectImage;
    else if (ui.toolButtonEdge->isChecked())
        saveMat = outputEdgeImage;
    else
        saveMat = inputImage;

    // ak je segmentovany objekt, pridaj anotaciu
    /*if (segmentedObjectArea > 0)
        saveMat = annotateImage(saveMat, segmentedObjectArea, imageArea);*/

    // zapis do suboru
    cv::imwrite(filename.toStdString(), saveMat);
}

// Ulozenie samostatneho segmentovaneho objektu
void ImageSegmentation::on_actionSaveObject_triggered()
{
    // skontroluj, ci existuje segmentacia
    if (inputImage.empty() || lastObjectMask.empty()) {
        QMessageBox::warning(this, "Warning", "No segmented object available to save");
        return;
    }

    // vytvor prazdny obrazok pozadia a skopiruj do neho objekt
    cv::Mat segmentedImage;
    if (lastIsLightObject)
        segmentedImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
    else
        segmentedImage = cv::Mat::ones(inputImage.size(), inputImage.type()) * 255;
    inputImage.copyTo(segmentedImage, lastObjectMask);

    // ak je objekt, pridaj anotaciu
    if (segmentedObjectArea > 0)
        segmentedImage = annotateImage(segmentedImage, segmentedObjectArea, imageArea);

    // vyber nazov suboru
    QString defaultName = loadedImageName.isEmpty() ? "segmented_object" : loadedImageName + "_object";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Segmented Object",
        defaultName + ".png",
        "PNG Image (*.png);;JPEG Image (*.jpg)");
    if (filename.isEmpty())
        return;

    // zapis do suboru
    cv::imwrite(filename.toStdString(), segmentedImage);
}

// Ulozenie vsetkych stavov segmentacie do priecinka
void ImageSegmentation::on_actionSaveAllStates_triggered()
{
    // vyber priecinka pre ukladanie
    QString folder = QFileDialog::getExistingDirectory(this,
        "Select Folder to Save All States",
        QDir::homePath());
    if (folder.isEmpty())
        return;

    // ziskanie prefixu pre nazvy suborov
    QString baseName = loadedImageName.isEmpty() ? "output" : loadedImageName;
    bool ok;
    QString customPrefix = QInputDialog::getText(this,
        "Save All States",
        "Enter file name prefix:",
        QLineEdit::Normal,
        baseName,
        &ok);
    if (ok && !customPrefix.isEmpty())
        baseName = customPrefix;

    // vytvorenie cest k suborom
    QString fileObject = folder + "/" + baseName + "_object.png";
    QString fileEdge = folder + "/" + baseName + "_edge.png";
    QString fileFeret = folder + "/" + baseName + "_feret.png";
    QString fileEllipse = folder + "/" + baseName + "_ellipse.png";
    QString fileMBR = folder + "/" + baseName + "_MBR.png";

    // uloz segmentovany objekt analogicky ako v on_actionSaveObject
    if (!inputImage.empty() && !lastObjectMask.empty())
    {
        cv::Mat segmentedImage;
        if (lastIsLightObject)
            segmentedImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
        else
            segmentedImage = cv::Mat::ones(inputImage.size(), inputImage.type()) * 255;
        inputImage.copyTo(segmentedImage, lastObjectMask);
        if (segmentedObjectArea > 0)
            segmentedImage = annotateImage(segmentedImage, segmentedObjectArea, imageArea);
        cv::imwrite(fileObject.toStdString(), segmentedImage);
    }
    else
    {
        // upozornenie, ak nie je co ulozit
        QMessageBox::warning(this, "Save All States", "No segmented object available to save.");
    }

    // uloz ostatne stavy
    if (!outputEdgeImage.empty())
        cv::imwrite(fileEdge.toStdString(), outputEdgeImage);
    if (!outputFeretImage.empty())
        cv::imwrite(fileFeret.toStdString(), outputFeretImage);
    if (!outputEllipseImage.empty())
        cv::imwrite(fileEllipse.toStdString(), outputEllipseImage);
    if (!outputMBRImage.empty())
        cv::imwrite(fileMBR.toStdString(), outputMBRImage);

    // informuj uzivatela o ukonceni
    QMessageBox::information(this,
        "Save All States",
        "All available state images have been saved to:\n" + folder);
}

// Ulozenie informacii o vybranej oblasti do textoveho suboru
void ImageSegmentation::on_actionSaveInfo_triggered()
{
    // skontroluj, ci mame nahraty obrazok a segmentacnu masku
    if (inputImage.empty() || lastObjectMask.empty()) {
        QMessageBox::warning(this, "Warning", "No segmented object available to compute info");
        return;
    }

    // ziskanie zakladnych statistickych udajov
    int area = segmentedObjectArea;
    double areaPercent = (imageArea > 0) ? (area * 100.0 / imageArea) : 0.0;
    double meanGray = cv::mean(inputImage, lastObjectMask)[0];

    // vypocet historamu intenzit
    cv::Mat hist;
    int histSize = 256;
    float range[] = { 0,256 };
    const float* histRange = { range };
    cv::calcHist(&inputImage, 1, 0, lastObjectMask, hist, 1, &histSize, &histRange);

    // najdenie modalnej hodnoty z histogrmu
    double maxHistVal = 0;
    int modalGray = 0;
    for (int i = 0; i < histSize; i++) {
        float hVal = hist.at<float>(i);
        if (hVal > maxHistVal) { maxHistVal = hVal; modalGray = i; }
    }

    // min a max intenzita v oblasti
    double minGray, maxGray;
    cv::minMaxLoc(inputImage, &minGray, &maxGray, nullptr, nullptr, lastObjectMask);

    // vypocet stredu hmotnosti (centroid)
    double sumX = 0, sumY = 0;
    int count = 0;
    for (int i = 0; i < lastObjectMask.rows; i++) {
        for (int j = 0; j < lastObjectMask.cols; j++) {
            if (lastObjectMask.at<uchar>(i, j) > 0) {
                sumX += j;
                sumY += i;
                count++;
            }
        }
    }
    double centroidX = (count > 0) ? (sumX / count) : 0;
    double centroidY = (count > 0) ? (sumY / count) : 0;

    // vypocet obvodu kontury
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(lastObjectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double perimeter = 0;
    if (!contours.empty()) {
        double maxContArea = 0;
        int idx = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > maxContArea) { maxContArea = a; idx = i; }
        }
        perimeter = cv::arcLength(contours[idx], true);
    }

    // vypocet Feretovych priemerov a priemeru z ekvivaletnej kruznice
    double longestFeret = 0.0, shortestFeret = 0.0, circleDiameter = 0.0;
    cv::Mat dummyAnnotated;
    computeFeretDiameterAndCircle(inputImage, lastObjectMask, dummyAnnotated, longestFeret, circleDiameter, shortestFeret);

    // vypocet hlavnych a vedlajsich osi Legendreovej elipsy
    double majorAxis = 0.0, minorAxis = 0.0;
    computeLegendreEllipse(inputImage, lastObjectMask, dummyAnnotated, majorAxis, minorAxis);

    // vypocet pomeru elipsy
    double ellipseRatio = (majorAxis != 0) ? (minorAxis / majorAxis) : 0.0;

    // vypocet rozmerov MBR a pomeru L/W
    double longDiameter = 0.0, shortDiameter = 0.0;
    computeMBR(inputImage, lastObjectMask, dummyAnnotated, longDiameter, shortDiameter);
    double LWRatio = (longDiameter != 0) ? (shortDiameter / longDiameter) : 0.0;

    // vypocet aspect ratio
    double aspectRatio = (longestFeret != 0) ? (shortestFeret / longestFeret) : 0.0;

    // vypocet circularity
    double perimEqDia = (perimeter > 0.0) ? (perimeter / M_PI) : 0.0;
    double circularity = (perimEqDia > 0.0) ? (circleDiameter / perimEqDia) : 0.0;

    // prepocty na nanometre
    double pSize = ui.doubleSpinBoxPixelSize->value();
    double area_nm2 = area * pSize;
    double perimeter_nm = perimeter * pSize;
    double longestFeret_nm = longestFeret * pSize;
    double shortestFeret_nm = shortestFeret * pSize;
    double majorAxis_nm = majorAxis * pSize;
    double minorAxis_nm = minorAxis * pSize;
    double longDiameter_nm = longDiameter * pSize;
    double shortDiameter_nm = shortDiameter * pSize;

    // skomponovanie retazca s informaciami
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

    // ulozenie do textoveho suboru
    QString defaultName = loadedImageName.isEmpty() ? "selection_info" : loadedImageName + "_info";
    QString filename = QFileDialog::getSaveFileName(this,
        "Save Selection Info",
        defaultName + ".txt",
        "Text Files (*.txt)");
    if (filename.isEmpty())
        return;
    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << info;
        file.close();
        QMessageBox::information(this, "Save Info", "Information saved successfully.");
    }
}

// Nastavenie mierky a zobrazenie
void ImageSegmentation::on_pushButtonScale_clicked()
{
    scaleFactor = ui.doubleSpinBoxScale->value();
    updateDisplay();
}

// Prepnutie zobrazenia na segmentovany objekt
void ImageSegmentation::on_toolButtonObject_toggled(bool checked)
{
    if (checked) resetToolButtons(ui.toolButtonObject);
    updateDisplay();
}

// Prepnutie zobrazenia na hrany
void ImageSegmentation::on_toolButtonEdge_toggled(bool checked)
{
    if (checked) resetToolButtons(ui.toolButtonEdge);
    updateDisplay();
}

// Prepnutie zobrazenia na Feretove priemery
void ImageSegmentation::on_toolButtonFeret_toggled(bool checked)
{
    if (checked) resetToolButtons(ui.toolButtonFeret);
    updateDisplay();
}

// Prepnutie zobrazenia na Legendreovu elipsu
void ImageSegmentation::on_toolButtonEllipse_toggled(bool checked)
{
    if (checked) resetToolButtons(ui.toolButtonEllipse);
    updateDisplay();
}

// Prepnutie zobrazenia na MBR
void ImageSegmentation::on_toolButtonMBR_toggled(bool checked)
{
    if (checked) resetToolButtons(ui.toolButtonMBR);
    updateDisplay();
}

// Aktivacia / ukoncenie rezimu pre ROI vyber
void ImageSegmentation::on_toolButtonSelectROIRectangle_toggled(bool checked)
{
    if (checked) {
        roiSelectionActive = true;
        roiFirstPointSet = false;
    }
    else {
        roiSelectionActive = false;
        roiFirstPointSet = false;
        updateSeedIntensities();
    }
}

void ImageSegmentation::on_toolButtonSelectROICustom_toggled(bool checked)
{
    if (checked) {
        polygonSelectionActive = true;
        roiPolygonPoints.clear();
        roiSelectionActive = false; // avoid conflict with rectangle mode
        ui.toolButtonSelectROIRectangle->setChecked(false);
    }
    else {
        polygonSelectionActive = false;
        if (!roiPolygonPoints.empty() && roiPolygonPoints.size() < 3) {
            roiPolygonPoints.clear(); // invalid polygon
        }
        updateSeedIntensities();
    }
}

// Spustenie segmentacie a zobrazenie vysledku
void ImageSegmentation::on_pushButtonProcess_clicked()
{
    if (inputImage.empty()) {
        QMessageBox::warning(this, "Warning", "Please open an image first");
        return;
    }
    runSegmentation();
    updateDisplay();
}