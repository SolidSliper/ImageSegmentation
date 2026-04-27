#include "BatchProcessor.h"
#include "ImageIO.h"

#include <QDir>
#include <QFileInfo>
#include <QDebug>

#include <opencv2/imgproc.hpp>

// ============================================================================
// Konstruktor / destruktor
// ============================================================================
BatchProcessor::BatchProcessor(QObject* parent)
    : QObject(parent)
{}

BatchProcessor::~BatchProcessor() = default;

// ============================================================================
// Praca s ONNX modelom
// ============================================================================

bool BatchProcessor::loadModel(const QString& modelPath)
{
    try {
        network = cv::dnn::readNetFromONNX(modelPath.toStdString());
        modelLoaded = !network.empty();
        return modelLoaded;
    }
    catch (const cv::Exception& e) {
        qWarning() << "Failed to load ONNX model:" << e.what();
        modelLoaded = false;
        return false;
    }
}

cv::Mat BatchProcessor::infer(const cv::Mat& input)
{
    if (!modelLoaded || input.empty())
        return cv::Mat();

    // Predspracovanie: normalizacia do <0,1>, NCHW blob.
    // Konkretne velkosti zavisia od architektury modelu - tu je default 256x256.
    constexpr int kInputSize = 256;

    cv::Mat blob = cv::dnn::blobFromImage(input,
                                          1.0 / 255.0,
                                          cv::Size(kInputSize, kInputSize),
                                          cv::Scalar(),
                                          /*swapRB=*/false,
                                          /*crop=*/false);
    network.setInput(blob);

    cv::Mat output;
    try {
        output = network.forward();
    }
    catch (const cv::Exception& e) {
        qWarning() << "Inference failed:" << e.what();
        return cv::Mat();
    }

    // Output blob: 1xCxHxW. Pre binarnu segmentaciu C=1, vyberme jediny kanal.
    cv::Mat probMap(output.size[2], output.size[3], CV_32F, output.ptr<float>(0, 0));

    // Resize spat na povodnu velkost vstupu.
    cv::Mat resized;
    cv::resize(probMap, resized, input.size(), 0, 0, cv::INTER_LINEAR);
    return resized.clone();
}

cv::Mat BatchProcessor::probMapToMask(const cv::Mat& probMap, float threshold)
{
    if (probMap.empty()) return cv::Mat();

    cv::Mat mask;
    cv::threshold(probMap, mask, threshold, 255.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);
    return mask;
}

void BatchProcessor::probMapToSeeds(const cv::Mat& input,
                                    const cv::Mat& probMap,
                                    int& objectSeed,
                                    int& backgroundSeed,
                                    float threshold)
{
    objectSeed = backgroundSeed = 0;
    if (input.empty() || probMap.empty()) return;

    long sumObj = 0, countObj = 0;
    long sumBack = 0, countBack = 0;

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float p = probMap.at<float>(i, j);
            int v = input.at<uchar>(i, j);
            if (p >= threshold) { sumObj  += v; countObj++;  }
            else                { sumBack += v; countBack++; }
        }
    }
    objectSeed     = (countObj  > 0) ? static_cast<int>(sumObj  / countObj)  : 0;
    backgroundSeed = (countBack > 0) ? static_cast<int>(sumBack / countBack) : 0;
}

// ============================================================================
// Davkove spracovanie
// ============================================================================

QStringList BatchProcessor::listImageFiles(const QString& directory)
{
    QDir dir(directory);
    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.jpeg" << "*.tif" << "*.tiff" << "*.bmp";
    return dir.entryList(filters, QDir::Files, QDir::Name);
}

int BatchProcessor::runBatch(const BatchConfig& config)
{
    // Pripravne kontroly
    QDir outDir(config.outputDirectory);
    if (!outDir.exists()) outDir.mkpath(".");

    if (config.useDeepLearning && !modelLoaded) {
        if (!loadModel(config.modelPath)) {
            qWarning() << "Cannot run DL batch - model not loaded.";
            emit batchFinished(0, 0);
            return 0;
        }
    }

    QStringList files = listImageFiles(config.inputDirectory);
    int total = files.size();
    int successCount = 0;

    for (int i = 0; i < total; ++i) {
        const QString& name = files[i];
        QString inPath  = config.inputDirectory  + "/" + name;
        QString outPath = config.outputDirectory + "/" + QFileInfo(name).baseName();

        bool ok = processSingleImage(inPath, outPath, config);
        if (ok) ++successCount;

        emit finishedFile(name, ok);
        emit progressChanged(i + 1, total);
    }

    emit batchFinished(successCount, total);
    return successCount;
}

bool BatchProcessor::processSingleImage(const QString& inputPath,
                                        const QString& outputPath,
                                        const BatchConfig& config)
{
    cv::Mat input = ImageIO::loadGrayscaleImage(inputPath);
    if (input.empty()) return false;

    Segmentation::Params params = config.segParams;

    // Volitelne: pouzi DL na odhad seed intenzit.
    if (config.useDeepLearning && modelLoaded) {
        cv::Mat probMap = infer(input);
        if (!probMap.empty()) {
            int objSeed = 0, bgSeed = 0;
            probMapToSeeds(input, probMap, objSeed, bgSeed);
            params.userObject     = objSeed;
            params.userBackground = bgSeed;

            // Ak nie je pozadovany graf-rez, vystupom je priamo DL maska.
            if (!config.refineWithGraphCut) {
                cv::Mat mask = probMapToMask(probMap);
                return ImageIO::saveImage(outputPath + "_mask.png", mask);
            }
        }
    }

    // Spusti graf-rez
    Segmentation::Result result;
    if (!Segmentation::segmentImage(input, params, result)) return false;

    // Ulozit vysledky
    bool ok = true;
    ok &= ImageIO::saveImage(outputPath + "_object.png", result.objectImage);
    ok &= ImageIO::saveImage(outputPath + "_edge.png",   result.edgeImage);
    ok &= ImageIO::saveImage(outputPath + "_mask.png",   result.objectMask);
    return ok;
}
