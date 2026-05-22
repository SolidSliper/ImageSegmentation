#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "Segmentation.h"

class BatchProcessor : public QObject
{
    Q_OBJECT

public:
    explicit BatchProcessor(QObject* parent = nullptr);
    ~BatchProcessor();
};
