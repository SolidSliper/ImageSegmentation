#include "ImageIO.h"

#include <opencv2/imgproc.hpp>

// ============================================================================
// Konverzie OpenCV <-> Qt
// ============================================================================

QImage ImageIO::cvMatToQImage(const cv::Mat& mat)
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
    default: {
        // preved na sivy
        cv::Mat grey;
        mat.convertTo(grey, CV_8U);
        QImage img(grey.data, grey.cols, grey.rows, grey.step, QImage::Format_Grayscale8);
        return img.copy();
    }
    }
}

cv::Mat ImageIO::QImageToCvMat(const QImage& image)
{
    if (image.isNull())
        return cv::Mat();

    switch (image.format()) {
    case QImage::Format_Grayscale8: {
        return cv::Mat(image.height(), image.width(), CV_8UC1,
                       const_cast<uchar*>(image.constBits()),
                       image.bytesPerLine()).clone();
    }
    case QImage::Format_RGB888: {
        cv::Mat tmp(image.height(), image.width(), CV_8UC3,
                    const_cast<uchar*>(image.constBits()),
                    image.bytesPerLine());
        cv::Mat mat;
        cv::cvtColor(tmp, mat, cv::COLOR_RGB2BGR);
        return mat;
    }
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    case QImage::Format_RGBA8888:
    case QImage::Format_RGBA8888_Premultiplied: {
        cv::Mat tmp(image.height(), image.width(), CV_8UC4,
                    const_cast<uchar*>(image.constBits()),
                    image.bytesPerLine());
        return tmp.clone();
    }
    default: {
        // Fallback: konvertuj na ARGB32
        QImage conv = image.convertToFormat(QImage::Format_ARGB32);
        cv::Mat tmp(conv.height(), conv.width(), CV_8UC4,
                    const_cast<uchar*>(conv.constBits()),
                    conv.bytesPerLine());
        return tmp.clone();
    }
    }
}

// ============================================================================
// Disk I/O
// ============================================================================

cv::Mat ImageIO::loadGrayscaleImage(const QString& filename)
{
    return cv::imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
}

bool ImageIO::saveImage(const QString& filename, const cv::Mat& image)
{
    if (image.empty()) return false;
    return cv::imwrite(filename.toStdString(), image);
}

// ============================================================================
// Pomocne operacie
// ============================================================================

cv::Mat ImageIO::removeInfoOverlay(const cv::Mat& input)
{
    int overlayHeight = input.rows / 9;
    cv::Rect cropRect(0, 0, input.cols, input.rows - overlayHeight);
    return input(cropRect).clone();
}
