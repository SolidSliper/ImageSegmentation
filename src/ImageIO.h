#pragma once

#include <opencv2/opencv.hpp>
#include <QImage>
#include <QString>

// ============================================================================
// ImageIO - I/O modul pre nacitavanie a ukladanie obrazov a konverzie
// medzi cv::Mat a QImage. Trieda nepotrebuje QObject - vsetky metody su
// staticke pomocne funkcie.
// ============================================================================
class ImageIO
{
public:
    // ------------------------------------------------------------------------
    // Konverzie medzi OpenCV a Qt
    // ------------------------------------------------------------------------

    // Prevod cv::Mat na QImage. Podporuje CV_8UC1 a CV_8UC3.
    // Vrati prazdny QImage ak je mat prazdny.
    static QImage cvMatToQImage(const cv::Mat& mat);

    // Prevod QImage na cv::Mat. Vrati kopiu (bez vlastnenia pamate QImage).
    static cv::Mat QImageToCvMat(const QImage& image);

    // ------------------------------------------------------------------------
    // Nacitavanie a ukladanie obrazov z disku
    // ------------------------------------------------------------------------

    // Nacita obraz zo suboru ako grayscale (CV_8UC1).
    // Vrati prazdny mat ak nacitanie zlyha.
    static cv::Mat loadGrayscaleImage(const QString& filename);

    // Ulozi obraz do suboru. Vrati true pri uspechu.
    static bool saveImage(const QString& filename, const cv::Mat& image);

    // ------------------------------------------------------------------------
    // Pomocne operacie nad obrazom
    // ------------------------------------------------------------------------

    // Odstrani spodny biely informacny overlay (typicke pre EM mikroskopiu).
    // Orez = 1/9 vysky obrazu zospodu.
    static cv::Mat removeInfoOverlay(const cv::Mat& input);
};
