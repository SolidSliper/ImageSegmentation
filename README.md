# ImageSegmentation

> Graph-based image segmentation tool for biomedical microscopy data — built on Dinic's max-flow / min-cut algorithm.

![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Qt](https://img.shields.io/badge/Qt-6.x-41CD52?logo=qt&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)

## About

`ImageSegmentation` is a desktop application for **precise segmentation and morphometric analysis of metallic micro/nano-particles** in scanning electron microscopy (SEM) images of human hippocampal tissue. It was developed as part of a bachelor's thesis at the Faculty of Civil Engineering, Slovak University of Technology in Bratislava, in collaboration with the Faculty of Medicine of Comenius University.

The application is being used to study the accumulation of iron (Fe), nickel (Ni), chromium (Cr) and zinc (Zn) particles in brain tissue — a topic increasingly linked to neurodegenerative diseases such as Alzheimer's and Parkinson's.

The segmentation engine reformulates the image as a flow network and computes the globally optimal binary cut using **Dinic's algorithm** (with **Edmonds–Karp** included for benchmarking). On top of the segmentation, the tool extracts a full set of geometric descriptors used in particle analysis.

## Key features

- **Two max-flow backends**: Dinic's algorithm (default) and Edmonds–Karp (for comparison).
- **Interactive Region of Interest** — rectangular or free-form polygon ROI to constrain the segmentation.
- **Light / Dark / Auto object modes** for objects brighter or darker than their background.
- **Tunable λ parameter** to balance regional vs. boundary terms in the energy function.
- **Post-processing**: BFS-based noise removal and hole filling on the resulting mask.
- **Geometric descriptors** computed automatically from the binary mask:
  - Projected area & perimeter
  - Equivalent area / perimeter diameters (Heywood diameter, circularity)
  - Min and max **Feret diameters**
  - **Legendre ellipse** (major/minor axes, aspect ratio)
  - **Minimum Bounding Rectangle (MBR)**
- **Pixel-to-nanometer calibration** for real physical units.
- **Batch mode** — process an entire folder of images in one click.
- **Export**: save segmentation states (original / binary / object / edge / Feret / ellipse / MBR) as images, and export numerical results to CSV.

## Achieved metrics

Validated on 12 SEM images of metallic particles, against reference masks manually annotated in **Fiji**:

| Metric             | Value                |
|--------------------|----------------------|
| Dice coefficient   | **0.9790 ± 0.0172**  |
| IoU (Jaccard)      | **0.9609 ± 0.0326**  |
| Pixel accuracy     | **0.9987 ± 0.0018**  |

### Algorithm benchmark

Same input image, identical segmentation parameters:

| Algorithm         | Segmentation time |
|-------------------|-------------------|
| Dinic             | **6.89 s**        |
| Edmonds–Karp      | 681.40 s          |

Dinic's algorithm is roughly **~100× faster** on real microscopy data, which is why it is the default in the application.

## Tech stack

- **C++17**
- **Qt 6** — UI (Qt Designer `.ui` file, signals/slots, `QElapsedTimer`)
- **OpenCV 4** — image I/O, `cv::Mat` containers, basic image operations
- Custom implementation of:
  - Graph construction from pixel grid (8-neighborhood, region + boundary terms)
  - Dinic's algorithm (level graph + blocking flows)
  - Edmonds–Karp algorithm (BFS-based augmenting paths)
  - Min-cut extraction
  - Connected-components post-processing (BFS, 4-neighborhood)
  - Geometric descriptors (Feret, Legendre ellipse, MBR)

## Project structure

```
src/
├── main.cpp                       # Entry point
├── ImageSegmentation.h            # Main class declaration
├── ImageSegmentation.cpp          # UI handling, I/O, ROI, Qt <-> OpenCV bridges
├── ImageSegmentationMath.cpp      # Pure compute module — graph, max-flow, geometry
├── ImageSegmentation.ui           # Qt Designer layout
└── ImageSegmentation.qrc          # Qt resources
```

## Roadmap

Goals are tracked in parallel rather than sequentially — none of them blocks the others.

- [x] Graph construction from grayscale images with regional + boundary energy terms
- [x] Dinic's max-flow / min-cut implementation
- [x] Edmonds–Karp implementation (for benchmarking)
- [x] Qt-based GUI with interactive parameter controls
- [x] Rectangular and free-form polygon ROI selection
- [x] Light / Dark / Auto object detection modes
- [x] Post-processing: noise removal + hole filling
- [x] Full geometric analysis (Feret, Legendre ellipse, MBR, area, perimeter, equivalent diameters)
- [x] Pixel → nanometer calibration
- [x] Batch folder processing
- [x] CSV export of measurements
- [x] Validation against manual Fiji masks (Dice 0.979, IoU 0.961)
- [ ] **Refactor the project** into proper modules.
- [ ] **Deep learning** for automated ROI marking — use a lightweight segmentation model to propose seed points and ROI automatically, removing the manual seeding step.
- [ ] **GPU parallelization** of the max-flow computation — the current bottleneck on high-resolution images is memory and the sequential nature of BFS phases. Investigate push-relabel variants suitable for CUDA / OpenCL.
- [ ] **First alpha release** — portable Windows build (no installer, just a zipped folder with the executable + Qt/OpenCV runtime DLLs).
- [ ] **Stable release with a proper Windows installer** (NSIS or Inno Setup).
- [ ] **Doxygen documentation** generated from the source comments, published as GitHub Pages.
- [ ] **UI overhaul** — cleaner layout, dark mode, better feedback during long-running segmentations, drag-and-drop image loading.
- [ ] **Pick a real product name** for the application (working title for now: `ImageSegmentation`).

### Possible future directions

- 3D segmentation for volumetric microscopy stacks
- Hybrid pipeline combining graph-cut precision with deep-learning priors
- Linux / macOS builds

## Building from source

> Detailed build instructions will be added with the first alpha release.

Currently the project is built in **Visual Studio 2022** with Qt and OpenCV configured via vcpkg. CMake support is on the way.

Required:
- Visual Studio 2022 (MSVC v143)
- Qt 6.x
- OpenCV 4.x

## Acknowledgments

- **Supervisor:** doc. Mgr. Mária Ždímalová, PhD. — Department of Mathematics and Descriptive Geometry, Faculty of Civil Engineering, STU Bratislava
- **Data provided by:** Department of Pathological Anatomy, Faculty of Medicine, Comenius University in Bratislava
- The graph-cut formulation follows the framework of Boykov & Kolmogorov; the max-flow implementation is based on Dinic's classical paper.

## License

To be decided before the first alpha release. Until then, the code is provided for review purposes only.

---

*This project started as a bachelor thesis ("Discrete Segmentation Algorithms in Biological and Medical Data Processing", STU Bratislava, 2026) and is being developed further into a standalone tool.*
