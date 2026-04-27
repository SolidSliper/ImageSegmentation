#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "Segmentation.h"

// ============================================================================
// BatchProcessor - davkove spracovanie obrazkov a integracia hlbokeho ucenia.
//
// Modul kombinuje dve oblasti:
//   1) Davkove spustenie segmentacie nad celym priecinkom obrazkov.
//   2) Inferenciu predtrenovaneho neuronoveho modelu (ONNX format) na
//      automaticke predspracovanie alebo ako alternativu k grafovemu rezu.
//
// Backend: cv::dnn (siet Net) - bez dalsich zavislosti.
// Trieda dedi z QObject aby mohla emitovat signaly o priebehu davky.
// ============================================================================
class BatchProcessor : public QObject
{
    Q_OBJECT

public:
    explicit BatchProcessor(QObject* parent = nullptr);
    ~BatchProcessor();

    // ------------------------------------------------------------------------
    // Konfiguracia davky
    // ------------------------------------------------------------------------
    struct BatchConfig {
        QString inputDirectory;       // priecinok so vstupnymi obrazmi
        QString outputDirectory;      // kam ukladat vysledky
        QString modelPath;            // cesta k ONNX modelu (volitelne)
        bool    useDeepLearning = false;  // ak true, pouzi DL na ziskanie seedov
        bool    refineWithGraphCut = true; // po DL este spustit graf-rez
        Segmentation::Params segParams;    // parametre grafoveho rezu
    };

    // ------------------------------------------------------------------------
    // Praca s ONNX modelom (cv::dnn)
    // ------------------------------------------------------------------------

    // Nacita ONNX model zo zadanej cesty. Vrati true pri uspechu.
    bool loadModel(const QString& modelPath);

    // Vykona inferenciu na jednom obraze a vrati pravdepodobnostnu mapu (CV_32F).
    // Predpoklada, ze model je nacitany cez loadModel().
    cv::Mat infer(const cv::Mat& input);

    // Prevod pravdepodobnostnej mapy na binarnu masku (255 / 0).
    static cv::Mat probMapToMask(const cv::Mat& probMap, float threshold = 0.5f);

    // Z pravdepodobnostnej mapy odvodi seed intenzity pre objekt a pozadie.
    // Vyuziva sa ako vstup pre Segmentation::segmentImage.
    static void probMapToSeeds(const cv::Mat& input,
                               const cv::Mat& probMap,
                               int& objectSeed,
                               int& backgroundSeed,
                               float threshold = 0.5f);

    // ------------------------------------------------------------------------
    // Davkove spracovanie
    // ------------------------------------------------------------------------

    // Spusti spracovanie celej davky podla konfiguracie.
    // Pre kazdy obraz emituje progressChanged a finishedFile.
    // Po skonceni emituje batchFinished.
    // Vrati pocet uspesne spracovanych suborov.
    int runBatch(const BatchConfig& config);

signals:
    // Priebezny stav davky: aktualny index a celkovy pocet.
    void progressChanged(int current, int total);

    // Emituje sa po dokonceni jedneho suboru (uspech alebo chyba).
    void finishedFile(const QString& filename, bool success);

    // Emituje sa po skonceni celej davky.
    void batchFinished(int successCount, int totalCount);

private:
    cv::dnn::Net network;     // nacitana siet (prazdna ak este nebola load-nuta)
    bool         modelLoaded = false;

    // Vrati zoznam podporovanych obrazov v priecinku (PNG, JPG, TIF).
    static QStringList listImageFiles(const QString& directory);

    // Spracuje jeden obraz podla konfiguracie. Vrati true pri uspechu.
    bool processSingleImage(const QString& inputPath,
                            const QString& outputPath,
                            const BatchConfig& config);
};
