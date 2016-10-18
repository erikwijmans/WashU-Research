#include "placeScan_doorDetector.h"
#include "placeScan_placeScan.h"
#include "placeScan_placeScanHelper.h"

#include <scan_gflags.h>

#include <Eigen/Geometry>

place::DoorDetector::DoorDetector()
    : loaded{false}, name{FLAGS_doorsFolder + "fpDoors.dat"} {
  if (fexists(name) && !FLAGS_redo) {
    std::ifstream in(name, std::ios::in | std::ios::binary);
    int length;
    in.read(reinterpret_cast<char *>(&length), sizeof(length));
    if (length == FLAGS_numLevels + 1) {
      loaded = true;
      responsePyr.resize(FLAGS_numLevels + 1);
      for (auto &r : responsePyr)
        loadSparseMatrix(r, in);
    }
    in.close();
  }
}

void place::DoorDetector::run(
    const std::vector<Eigen::SparseMatrix<double>> &fpPyramid,
    const std::vector<Eigen::SparseMatrix<double>> &erodedFpPyramid,
    const std::vector<Eigen::MatrixXb> &fpMasks) {

  if (loaded)
    return;

  symbols.resize(NUM_ROTS * 2);
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

  std::vector<Eigen::Vector2d> toDoor(NUM_ROTS * 2);
  toDoor[0] = Eigen::Vector2d::UnitX();

  for (int k = 0; k < NUM_ROTS; ++k) {
    cv::Mat rotMat = cv::getRotationMatrix2D(
        cv::Point2f(widenedDoor.cols / 2.0, widenedDoor.rows / 2.0), k * 90, 1);

    Eigen::Rotation2D<double> R(-k * M_PI / 2.0);
    toDoor[2 * k] = R * toDoor[0];

    cv::Mat tmp;
    cv::warpAffine(widenedDoor, tmp, rotMat, widenedDoor.size());
    int maxRow = 0, minRow = tmp.rows;
    int maxCol = 0, minCol = tmp.cols;
    for (int j = 5; j < tmp.rows - 5; ++j) {
      auto src = tmp.ptr<uchar>(j);
      for (int i = 5; i < tmp.cols - 5; ++i) {
        if (src[i] == 0) {
          maxRow = std::max(j, maxRow);
          minRow = std::min(j, minRow);

          maxCol = std::max(i, maxCol);
          minCol = std::min(i, minCol);
        }
      }
    }

    symbols[2 * k] =
        cv::Mat(maxRow - minRow + 1, maxCol - minCol + 1, doorSymbol.type());
    for (int j = minRow; j < maxRow + 1; ++j) {
      auto src = tmp.ptr<uchar>(j);
      auto dst = symbols[2 * k].ptr<uchar>(j - minRow);
      for (int i = minCol; i < maxCol + 1; ++i) {
        dst[i - minCol] = src[i];
      }
    }

    cv::flip(symbols[2 * k], symbols[2 * k + 1], 1);

    toDoor[2 * k + 1] = toDoor[2 * k];
    toDoor[2 * k + 1][0] *= -1.0;

    for (int i = 0; i < 2 && FLAGS_previewOut; ++i) {
      cv::Mat_<uchar> tmp = symbols[2 * k + i].clone();
      const auto &d = toDoor[2 * k + i];
      for (int j = 0; j < tmp.rows / 4.0; ++j)
        tmp(tmp.rows / 2.0 + d[1] * j, tmp.cols / 2.0 + d[0] * j) =
            255 - j * 255.0 / (tmp.rows / 4.0);

      cv::rectshow(tmp);
    }
  }

  std::vector<Eigen::SparseMatrix<double>> sparseSymbols;
  for (auto &s : symbols)
    sparseSymbols.emplace_back(place::scanToSparse(s));

  std::vector<std::vector<Eigen::SparseMatrix<double>>> symbolPyr(
      {sparseSymbols});
  place::createPyramid(symbolPyr, levels);

  std::vector<std::vector<Eigen::MatrixXb>> masks;
  for (int i = 0; i < levels + 1; ++i) {
    std::vector<Eigen::MatrixXb> tmp;
    for (auto &s : symbolPyr[i])
      tmp.emplace_back(Eigen::MatrixXb::Ones(s.rows(), s.cols()));
    masks.emplace_back(tmp);
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
    Eigen::MatrixXb doors =
        Eigen::MatrixXb::Zero(fpPyramid[k].rows(), fpPyramid[k].cols());
    std::vector<std::vector<place::Door>> pcDoors(NUM_ROTS * 2);
    findPlacement(fpPyramid[k], symbolPyr[k], erodedFpPyramid[k], symbolPyr[k],
                  masks[k], numPixelsUnderMask[k], fpMasks[k], pointsToAnalyze,
                  doors, pcDoors, scores);
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

  double average, sigma;
  std::tie(average, sigma) = place::aveAndStdev(
      minima.begin(), minima.end(), [](auto &m) { return m->score; });

  cv::Mat_<cv::Vec3b> out = fpColor.clone();
  Eigen::RowMatrixXb denseMap =
      Eigen::RowMatrixXb::Zero(fpPyramid[0].rows(), fpPyramid[0].cols());

  for (int j = 0; j < denseMap.rows(); ++j)
    for (int i = 0; i < denseMap.cols(); ++i)
      if (out(j, i) == cv::Vec3b(255, 255, 255))
        denseMap(j, i) = 1;

  for (auto &min : minima) {
    if (min->score >= (average - 1.2 * sigma))
      continue;

    auto &img = symbols[min->rotation];
    auto offset = toDoor[min->rotation];
    offset[0] *= img.cols / 2.0;
    offset[1] *= img.rows / 2.0;

    const int rows = img.rows;
    const int cols = img.cols;

    for (int j = 0; j < rows; ++j) {
      for (int i = 0; i < cols; ++i) {
        out(j + min->y + offset[1], i + min->x + offset[0]) =
            cv::Vec3b(0, 0, 255);
        denseMap(j + min->y + offset[1], i + min->x + offset[0]) = 2;
      }
    }
  }

  Eigen::SparseMatrix<char> response = denseMap.sparseView();

  responsePyr.assign({response});
  place::createPyramid(responsePyr, FLAGS_numLevels);

  if (FLAGS_save) {
    std::ofstream binaryWriter(name, std::ios::out | std::ios::binary);
    int length = responsePyr.size();
    binaryWriter.write(reinterpret_cast<const char *>(&length), sizeof(length));
    for (auto &r : responsePyr)
      saveSparseMatrix(r, binaryWriter);
    binaryWriter.close();
  }

  if (FLAGS_previewOut)
    cv::rectshow(out);

  loaded = true;
}

const Eigen::SparseMatrix<char> &
place::DoorDetector::getResponse(int level) const {
  return responsePyr[level];
}
