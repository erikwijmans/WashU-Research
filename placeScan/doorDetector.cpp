#include "placeScan_doorDetector.h"
#include "placeScan_placeScanHelper.h"

#include <scan_gflags.h>

place::DoorDetector::DoorDetector(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::vector<Eigen::MatrixXb> &fpMasks)
    : response{floorPlan.size(), 255}, symbols{NUM_ROTS * 2} {

  constexpr int levels = 3;
  const cv::Mat doorSymbol = cv::imread(FLAGS_dataPath + "/doorSymbol.png", 0);
  if (!doorSymbol.data) {
    std::cout << "Error loading door symbol!" << std::endl;
    exit(1);
  }

  int rows = doorSymbol.rows;
  int cols = doorSymbol.cols;

  int newRows, newCols;
  newRows = std::max(rows, cols) + 50;
  newCols = newRows;
  int rowDelta = (newRows - rows) / 2.0, colDelta = (newCols - cols) / 2.0;

  cv::Mat widenedDoor(newRows, newCols, doorSymbol.type(),
                      cv::Scalar::all(255));

  for (int j = 0; j < doorSymbol.rows; ++j) {
    auto src = doorSymbol.ptr<uchar>(j);
    auto dst = widenedDoor.ptr<uchar>(j + rowDelta);
    for (int i = 0; i < doorSymbol.cols; ++i) {
      dst[i + colDelta] = src[i];
    }
  }

  for (auto &s : symbols)
    s = cv::Mat(doorSymbol.size() + cv::Size(5, 5), doorSymbol.type());

  for (int k = 0; k < NUM_ROTS; ++k) {
    cv::Mat rotMat = cv::getRotationMatrix2D(
        cv::Point2f(widenedDoor.cols / 2.0, widenedDoor.rows / 2.0), k * 90, 1);

    cv::Mat tmp;
    cv::warpAffine(widenedDoor, tmp, rotMat, widenedDoor.size());
    for (int j = 0; j < doorSymbol.rows + 5; ++j) {
      auto dst = symbols[2 * k].ptr<uchar>(j);
      auto src = tmp.ptr<uchar>(j + rowDelta - 2);
      for (int i = 0; i < doorSymbol.cols + 5; ++i) {
        dst[i] = src[i + rowDelta - 2];
      }
    }
    cv::flip(symbols[2 * k], symbols[2 * k + 1], 1);
  }

  std::vector<Eigen::SparseMatrix<double>> sparseSymbols;
  for (auto &s : symbols)
    sparseSymbols.emplace_back(place::scanToSparse(s));

  std::vector<std::vector<Eigen::SparseMatrix<double>>> symbolPyr(
      {sparseSymbols});
  place::createPyramid(symbolPyr, levels);

  std::vector<std::vector<Eigen::MatrixXb>> masks;
  for (int i = 0; i < levels + 1; ++i) {
    masks.emplace_back(std::vector<Eigen::MatrixXb>(
        symbolPyr[i].size(),
        Eigen::MatrixXb::Ones(symbolPyr[i][0].rows(), symbolPyr[i][0].cols())));
  }

  std::vector<Eigen::VectorXd> numPixelsUnderMask;
  place::findNumPixelsUnderMask(symbolPyr, masks, numPixelsUnderMask);

  constexpr double numRects = 256;

  std::vector<place::posInfo> scores;
  std::vector<const posInfo *> minima;
  std::vector<Eigen::Vector3i> pointsToAnalyze;
  /*
  * Initializ pointsToAnalyze with every point
  */
  pointsToAnalyze.reserve(2 * NUM_ROTS * fpPyramid[levels].cols() *
                          fpPyramid[levels].rows());
  for (int k = 0; k < 2 * NUM_ROTS; ++k) {
    const int xStop = fpPyramid[levels].cols() - symbolPyr[levels][k].cols();

    const int yStop = fpPyramid[levels].rows() - symbolPyr[levels][k].rows();

    for (int i = 0; i < xStop; ++i)
      for (int j = 0; j < yStop; ++j)
        pointsToAnalyze.push_back(Eigen::Vector3i(i, j, k));
  }
  pointsToAnalyze.shrink_to_fit();

  /*
  * Main work loop.  This takes care of doing all the method calls
  * needed to make pryamiding work.  Each method will take of making sure
  * the container passed to it for it's output is cleared
  */
  for (int k = levels; k >= 0; --k) {
    findPlacement(fpPyramid[k], symbolPyr[k], erodedFpPyramid[k], symbolPyr[k],
                  masks[k], numPixelsUnderMask[k], fpMasks[k], pointsToAnalyze,
                  scores);
    if (scores.size() == 0)
      return;

    const int scanRows =
        std::min({symbolPyr[k][0].rows(), symbolPyr[k][1].rows(),
                  symbolPyr[k][2].rows(), symbolPyr[k][3].rows()});
    const int scanCols =
        std::min({symbolPyr[k][0].cols(), symbolPyr[k][1].cols(),
                  symbolPyr[k][2].cols(), symbolPyr[k][3].cols()});

    const double exclusion =
        (scanRows + scanCols) / (2.0 * std::sqrt(numRects));

    const int cols = fpPyramid[k].cols();
    const int rows = fpPyramid[k].rows();
    place::ExclusionMap maps(exclusion, rows, cols, 8);

    if (k == 0)
      findLocalMinima(scores, -0.5, maps, minima);
    else
      findLocalMinima(scores, 1.2, maps, minima);

    findPointsToAnalyzeV2(minima, pointsToAnalyze);
  }

  std::sort(minima.begin(), minima.end(),
            [](const place::posInfo *a, const place::posInfo *b) {
              return (a->score < b->score);
            });

  if (FLAGS_visulization || FLAGS_previewOut)
    place::displayOutput(fpPyramid[0], symbolPyr[0], minima);

  for (auto &min : minima)
    response(min->y, min->x) = min->score;

  cv::Mat_<cv::Vec3b> out = fpColor.clone();
  for (int j = 0; j < out.rows; ++j) {
    for (int i = 0; i < out.cols; ++i) {
      if (response(j, i) < 0.1) {
        out(cv::Range(j - 10, j + 10), cv::Range(i - 10, i + 10)) =
            cv::Vec3b(0, 255, 0);
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", out);
  cv::waitKey(0);
}