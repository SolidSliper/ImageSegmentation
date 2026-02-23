#pragma once

#include <QtWidgets/QMainWindow>
#include <QScrollArea>
#include <QLabel>
#include <QButtonGroup>
#include "ui_ImageSegmentation.h"

#include <opencv2/opencv.hpp>

// Hlavna trieda pre segmentaciu obrazkov
class ImageSegmentation : public QMainWindow
{
    Q_OBJECT

public:
    // Konstruktor a destruktor
    ImageSegmentation(QWidget* parent = nullptr);
    ~ImageSegmentation();

    // Vnorene triedy grafu
    class Node;
    class Link;
    class Graph;

    // Rezimy segmentacie: Light (svetly objekt), Dark (tmavy objekt), Auto (automaticky)
    enum class SegmentationMode { Light, Dark, Auto };

    // Hlavna segmentacna funkcia
    // input: vstupny obraz
    // mode: rezim segmentacie
    // lambda: regulacny parameter
    // outputObject: vystupny obraz objektu
    // outputEdges: vystupny obraz hran
    // objectMask: binarna maska objektu
    // isLightObject: flag, ci je objekt svetly
    // roiMask: volitelna ROI maska, ak je prazdna, vypocita sa automaticky
    bool segmentImage(const cv::Mat& input,
        SegmentationMode mode,
        double lambda,
        cv::Mat& outputObject,
        cv::Mat& outputEdges,
        cv::Mat& objectMask,
        bool& isLightObject,
        const cv::Mat& roiMask = cv::Mat());

protected:
    // Prepisane pre zachytavanie mysovych udalosti pri vybere ROI
    bool eventFilter(QObject* obj, QEvent* event) override;

private:
    Ui::ImageSegmentationClass ui;  // Rozhranie z .ui suboru

    // Obrazove data
    cv::Mat inputImage;             // Vstupny obraz v sivej stupnici
    cv::Mat outputObjectImage;      // Obraz s prekryvmi objektu
    cv::Mat outputEdgeImage;        // Obraz s prekryvmi hran
    cv::Mat lastObjectMask;         // Posledna segmentacna maska
    bool lastIsLightObject = true;  // Posledna hodnota rezimu
    bool polygonSelectionActive = false;
    std::vector<cv::Point> roiPolygonPoints;

    // Ulozene prekryvne obrazky vypocitane pri segmentacii
    cv::Mat outputFeretImage;
    cv::Mat outputEllipseImage;
    cv::Mat outputMBRImage;

    // Zobrazenie obrazka (prevedeny na QImage)
    QImage currentDisplay;

    // Ovladanie rezimu segmentacie
    SegmentationMode currentMode = SegmentationMode::Light;
    QButtonGroup modeButtonGroup;

    // Parametre algoritmu
    double scaleFactor = 0.6;       // Zmena mierky zobrazenia
    double lambda = 1;              // Regulacny parameter
    double pixelSizeNm;             // Velkost pixelu pre kalibraciu
    int defaultObjectIntensity = 0; // Vychodzia intenzita objektu
    int defaultBackgroundIntensity = 0; // Vychodzia intenzita pozadia
    int threshold = 100;            // Prahova hodnota (volitelna)
    int imageArea;                  // Plocha obrazka v pixelov
    int segmentedObjectArea = 0;    // Plocha segmentovaneho objektu
    int userObject, userBackground; // Intenzity nastavene uzivatelom

    // ROI maska: ak je neprazdna, pouzije sa na obmedzenie segmentacie
    cv::Mat userROIMask;

    // Zakladne meno nacitaneho obrazka pre ukladanie
    QString loadedImageName;

    // Zvoli spravnu ROI masku (uzivatel alebo automaticky)
    cv::Mat applyROIMask(const cv::Mat& input);

    // Vycisti vsetky interni data pri otvoreni noveho obrazka
    void clearAllData();

    // Detekuje a odstrani biely info overlay na spodku obrazka
    cv::Mat removeInfoOverlay(const cv::Mat& input);

    // Segmentacia a tvorba prekryvov
    void runSegmentation();
    void displayImage(const QImage& img);
    QImage cvMatToQImage(const cv::Mat& mat);

    // Vypocet geometrickych parametrov zo segmentacnej masky
    void computeFeretDiameterAndCircle(const cv::Mat& input,
        const cv::Mat& objectMask,
        cv::Mat& annotatedImage,
        double& longestFeret,
        double& circleDiameter,
        double& shortestFeret);

    void computeLegendreEllipse(const cv::Mat& input,
        const cv::Mat& objectMask,
        cv::Mat& annotatedImage,
        double& majorAxis,
        double& minorAxis);

    void computeMBR(const cv::Mat& input,
        const cv::Mat& objectMask,
        cv::Mat& annotatedImage,
        double& longDiameter,
        double& shortDiameter);

    // Aktualizuje seed intenzity podla ROI masky
    void updateSeedIntensities();

    void computeSeedIntensities(const cv::Mat& inputImage,
        ImageSegmentation::SegmentationMode mode,
        int& defaultObjectIntensity,
        int& defaultBackgroundIntensity,
        const cv::Mat& roiMask = cv::Mat());
    cv::Mat annotateImage(const cv::Mat& img, double areaObject, int areaImage);

    // Pre interaktivny vyber ROI
    bool roiSelectionActive = false;  // Ci je aktivny rezim vyberu
    bool roiFirstPointSet = false;    // Ci bol nastavany prvy bod
    cv::Point roiFirstPoint, roiSecondPoint; // Koordinaty vyberu
    void resetToolButtons(QAbstractButton* active);

private slots:
    // Akcie suboroveho menu
    void on_actionOpen_triggered();        // Otvorenie obrazka
    void on_actionSave_triggered();        // Ulozi obrazok s prekryvmi
    void on_actionSaveObject_triggered();  // Ulozi segmentovany objekt
    void on_actionSaveInfo_triggered();    // Ulozi statistiku segmentacie
    void on_actionSaveAllStates_triggered();  // Ulozi vsetky medzistavy

    // Spracovanie obrazu
    void on_pushButtonProcess_clicked();   // Spusti segmentaciu
    void on_pushButtonScale_clicked();     // Zmeni mierku obrazu

    // Tlacidla pre prepnutie zobrazenia prekryvov
    void updateDisplay();
    void on_toolButtonFeret_toggled(bool checked);
    void on_toolButtonObject_toggled(bool checked);
    void on_toolButtonEdge_toggled(bool checked);
    void on_toolButtonEllipse_toggled(bool checked);
    void on_toolButtonMBR_toggled(bool checked);

    // Tlacidlo pre interaktivny vyber ROI
    void on_toolButtonSelectROIRectangle_toggled(bool checked);
    void on_toolButtonSelectROICustom_toggled(bool checked);
};