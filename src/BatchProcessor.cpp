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