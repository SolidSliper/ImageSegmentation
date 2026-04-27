#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// ============================================================================
// Segmentation - vypoctove jadro segmentacie obrazu pomocou grafoveho rezu.
// Tato trieda je nezavisla od Qt a uzivatelskeho rozhrania. Obsahuje:
//   - reprezentaciu grafu (Node, Link, Graph),
//   - Dinicov a Edmondsov-Karpov algoritmus pre maximalny tok,
//   - extrakciu minimalneho rezu,
//   - cistenie masky (sum, diery),
//   - geometricke analyzy (Feret, Legendreova elipsa, MBR),
//   - automaticky vypocet seed intenzit.
//
// Vsetky metody su staticke - trieda sluzi ako menny priestor.
// ============================================================================
class Segmentation
{
public:
    // ------------------------------------------------------------------------
    // Rezimy segmentacie
    // ------------------------------------------------------------------------
    enum class Mode { Light, Dark, Auto };

    // Pouzity algoritmus pre vypocet maximalneho toku
    enum class Algorithm { Dinic, EdmondsKarp };

    // ------------------------------------------------------------------------
    // Vstupne parametre segmentacie - nahradzaju globalny stav z UI.
    // Volajuci (UI alebo BatchProcessor) ich naplni pred volanim segmentImage.
    // ------------------------------------------------------------------------
    struct Params {
        Mode      mode                = Mode::Light;
        Algorithm algorithm           = Algorithm::Dinic;
        double    lambda              = 1.0;   // regulacny parameter T-liniek
        int       userObject          = 0;     // intenzita seed pre objekt
        int       userBackground      = 0;     // intenzita seed pre pozadie
        int       noiseThreshold      = 100;   // min. velkost komponentu
        int       holeSize            = 0;     // max. velkost zaplnanej diery
    };

    // ------------------------------------------------------------------------
    // Vystup segmentacie
    // ------------------------------------------------------------------------
    struct Result {
        cv::Mat objectImage;    // farebny prekryv objektu
        cv::Mat edgeImage;      // farebny prekryv hran
        cv::Mat objectMask;     // binarna maska (255 = objekt)
        bool    isLightObject;  // skutocny rezim po Auto-detekcii
    };

    // ------------------------------------------------------------------------
    // Vnorene triedy grafu - reprezentacia siete pre max-flow.
    // ------------------------------------------------------------------------
    class Node;
    class Link;
    class Graph;

    // ------------------------------------------------------------------------
    // Hlavne API
    // ------------------------------------------------------------------------

    // Hlavna segmentacna funkcia.
    //  input    - vstupny grayscale obraz (CV_8UC1).
    //  params   - parametre segmentacie.
    //  result   - vystupne obrazy a maska (zapise sa).
    //  roiMask  - volitelna ROI maska, ak je prazdna, segmentuje sa cely obraz.
    // Vrati true pri uspechu.
    static bool segmentImage(const cv::Mat& input,
                             const Params& params,
                             Result& result,
                             const cv::Mat& roiMask = cv::Mat());

    // Automaticky vypocet seed intenzit z histogramu (s prihliadnutim na ROI).
    static void computeSeedIntensities(const cv::Mat& inputImage,
                                       Mode mode,
                                       int& objectIntensity,
                                       int& backgroundIntensity,
                                       const cv::Mat& roiMask = cv::Mat());

    // ------------------------------------------------------------------------
    // Geometricke parametre - vsetky pracuju s binarnou maskou
    // ------------------------------------------------------------------------
    static void computeFeretDiameterAndCircle(const cv::Mat& input,
                                              const cv::Mat& objectMask,
                                              cv::Mat& annotatedImage,
                                              double& longestFeret,
                                              double& circleDiameter,
                                              double& shortestFeret);

    static void computeLegendreEllipse(const cv::Mat& input,
                                       const cv::Mat& objectMask,
                                       cv::Mat& annotatedImage,
                                       double& majorAxis,
                                       double& minorAxis);

    static void computeMBR(const cv::Mat& input,
                           const cv::Mat& objectMask,
                           cv::Mat& annotatedImage,
                           double& longDiameter,
                           double& shortDiameter);
};

// ============================================================================
// Definicie vnorenych tried grafu.
// Su deklarovane mimo tela triedy aby ostal main header citatelny.
// ============================================================================
class Segmentation::Node
{
public:
    int id;
    int intensity;          // pre pixelove uzly
    int x, y;               // suradnice pixelu (-1 pre source/sink)
    std::vector<Link*> adj;

    Node(int id, int intensity = 0, int x = -1, int y = -1)
        : id(id), intensity(intensity), x(x), y(y) {}
};

class Segmentation::Link
{
public:
    double capacity;
    double flow;
    Node*  to;
    Link*  reverse;

    Link(double cap, Node* toNode)
        : capacity(cap), flow(0), to(toNode), reverse(nullptr) {}
};

class Segmentation::Graph
{
public:
    std::vector<Node*> nodes;
    Node* source;
    Node* sink;

    ~Graph() {
        for (Node* node : nodes) {
            for (Link* edge : node->adj)
                delete edge;
            delete node;
        }
    }
};
