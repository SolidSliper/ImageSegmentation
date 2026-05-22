#pragma once

#include <QtWidgets/QMainWindow>
#include <QButtonGroup>
#include "ui_ImageSegmentation.h"

#include <opencv2/opencv.hpp>

#include "Segmentation.h"

// ============================================================================
// ImageSegmentation - hlavne okno aplikacie.
//
// Zodpoveda iba za uzivatelske rozhranie:
//   - reakcie na stlacenia tlacidiel a menu polozky,
//   - vyber ROI (myskou),
//   - zobrazenie obrazkov v QLabel,
//   - orchestraciu volani do modulov Segmentation, ImageIO, BatchProcessor.
//
// Vsetka vypoctova logika a I/O operacie boli presunute do samostatnych
// modulov - tato trieda len napaja UI na ich verejne API.
// ============================================================================
class ImageSegmentation : public QMainWindow
{
    Q_OBJECT

public:
    ImageSegmentation(QWidget* parent = nullptr);
    ~ImageSegmentation();

protected:
    // Zachytava mysove udalosti pri vybere ROI.
    bool eventFilter(QObject* obj, QEvent* event) override;

private:
    Ui::ImageSegmentationClass ui;

    // ------------------------------------------------------------------------
    // Stav obrazov
    // ------------------------------------------------------------------------
    cv::Mat inputImage;
    cv::Mat outputObjectImage;
    cv::Mat outputEdgeImage;
    cv::Mat outputFeretImage;
    cv::Mat outputEllipseImage;
    cv::Mat outputMBRImage;
    cv::Mat lastObjectMask;
    cv::Mat userROIMask;
    QImage  currentDisplay;

    // ------------------------------------------------------------------------
    // Stav UI a parametre
    // ------------------------------------------------------------------------
    Segmentation::Mode currentMode = Segmentation::Mode::Light;
    QButtonGroup       modeButtonGroup;

    bool   lastIsLightObject     = true;
    double scaleFactor           = 0.6;
    double lambda                = 1.0;
    double pixelSizeNm           = 0.0;
    int    defaultObjectIntensity     = 0;
    int    defaultBackgroundIntensity = 0;
    int    threshold             = 100;
    int    imageArea             = 0;
    int    segmentedObjectArea   = 0;
    int    userObject            = 0;
    int    userBackground        = 0;

    QString loadedImageName;

    // ROI - obdlznik
    bool       roiSelectionActive = false;
    bool       roiFirstPointSet   = false;
    cv::Point  roiFirstPoint;
    cv::Point  roiSecondPoint;

    // ROI - polygon
    bool                   polygonSelectionActive = false;
    std::vector<cv::Point> roiPolygonPoints;

    // ------------------------------------------------------------------------
    // Pomocne metody UI
    // ------------------------------------------------------------------------

    // Vyresetuje vsetky obrazy a stav pri otvoreni noveho suboru.
    void clearAllData();
    void clearImageData();

    // Vrati ROI masku - pouziva uzivatelsku ak existuje, inak orez overlay.
    cv::Mat applyROIMask(const cv::Mat& input);

    // Zobrazi QImage v ui.imageLabel s aktualnym scaleFactor.
    void displayImage(const QImage& img);

    // Spusti segmentaciu cez Segmentation::segmentImage a aktualizuje UI.
    void runSegmentation();

    // Prepocita seed intenzity podla aktualneho rezimu a ROI a aktualizuje UI.
    void updateSeedIntensities();

    // Postavi Segmentation::Params zo stavu UI.
    Segmentation::Params buildParams() const;

private slots:
    // Subor
    void on_actionOpen_triggered();
    void on_actionSave_triggered();
    void on_actionSaveObject_triggered();
    void on_actionSaveInfo_triggered();
    void on_actionSaveAllStates_triggered();
    void on_actionProcessFolder_triggered();

    // Spracovanie
    void on_pushButtonProcess_clicked();
    void on_pushButtonScale_clicked();

    // Prepinace zobrazenia
    void on_actionOriginal_triggered();
    void on_actionFeret_triggered();
    void on_actionEdge_triggered();
    void on_actionObject_triggered();
    void on_actionEllipse_triggered();
    void on_actionMBR_triggered();

    // Slider <-> SpinBox synchronizacia
    void on_horizontalSliderBackground_valueChanged();
    void on_horizontalSliderObject_valueChanged();
    void on_doubleSpinBoxBackground_valueChanged();
    void on_doubleSpinBoxObject_valueChanged();

    // ROI
    void on_toolButtonSelectROIRectangle_toggled(bool checked);
    void on_toolButtonSelectROICustom_toggled(bool checked);
};
