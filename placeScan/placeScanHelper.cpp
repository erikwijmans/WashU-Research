#include "placeScan_placeScanHelper.h"

#include <dirent.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat fpColor, floorPlan;
std::vector<Eigen::Vector3i> truePlacement;
const double maxDelta = 0.10, maxTotal = 0.15;

static constexpr int minWrite = 200;

void place::parseFolders(std::vector<std::string> &pointFileNames,
                         std::vector<std::string> &zerosFileNames,
                         std::vector<std::string> *freeFileNames) {

  const std::string newDmFolder = FLAGS_dmFolder + "R0/";
  parseFolder(newDmFolder, pointFileNames, [](const std::string &s) {
    return s.find("point") != std::string::npos;
  });
  if (freeFileNames) {
    parseFolder(newDmFolder, *freeFileNames, [](const std::string &s) {
      return s.find("freeSpace") != std::string::npos;
    });
  }
  const std::string zzFolder = FLAGS_zerosFolder;
  parseFolder(zzFolder, zerosFileNames);

  if (pointFileNames.size() != zerosFileNames.size()) {
    perror("Not the same number of scans as zeros!");
    exit(1);
  }
}

void place::loadInScans(const std::string &scanName,
                        const std::string &zerosFile,
                        std::vector<cv::Mat> &rotatedScans,
                        std::vector<Eigen::Vector2i> &zeroZero) {

  zeroZero.resize(NUM_ROTS);
  std::ifstream binaryReader(zerosFile, std::ios::in | std::ios::binary);
  for (auto &z : zeroZero)
    binaryReader.read(reinterpret_cast<char *>(z.data()),
                      sizeof(Eigen::Vector2i));

  binaryReader.close();

  for (int i = 0; i < NUM_ROTS; ++i) {
    std::string fullScanName =
        FLAGS_dmFolder + "R" + std::to_string(i) + "/" + scanName;

    rotatedScans.push_back(cv::imread(fullScanName, 0));

    if (!rotatedScans[i].data) {
      std::cout << "Error reading scan: " << fullScanName << std::endl;
      exit(1);
    }
  }

  if (FLAGS_tinyPreviewIn || FLAGS_visulization) {
    cv::rectshow(rotatedScans[0]);
  }
}

void place::loadInScans(const std::string &scanName,
                        std::vector<cv::Mat> &rotatedScans) {

  for (int i = 0; i < NUM_ROTS; ++i) {
    std::string fullScanName =
        FLAGS_dmFolder + "R" + std::to_string(i) + "/" + scanName;

    rotatedScans.push_back(cv::imread(fullScanName, 0));

    if (!rotatedScans[i].data) {
      std::cout << "Error reading scan" << std::endl;
      exit(1);
    }
  }

  if (FLAGS_tinyPreviewIn || FLAGS_visulization) {
    cv::rectshow(rotatedScans[0]);
  }
}

void place::loadInScansAndMasks(const std::string &scanName,
                                const std::string &zerosFile,
                                const std::string &maskName,
                                std::vector<cv::Mat> &rotatedScans,
                                std::vector<cv::Mat> &masks,
                                std::vector<Eigen::Vector2i> &zeroZero) {

  place::loadInScans(scanName, zerosFile, rotatedScans, zeroZero);
  place::loadInScans(maskName, masks);
}

void place::trimScans(const std::vector<cv::Mat> &toTrim,
                      std::vector<cv::Mat> &trimmedScans,
                      std::vector<Eigen::Vector2i> &zeroZero) {
  int k = 0;
  for (auto &scan : toTrim) {
    int minRow = scan.rows;
    int minCol = scan.cols;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = 0; i < scan.rows; ++i) {
      const uchar *src = scan.ptr<uchar>(i);
      for (int j = 0; j < scan.cols; ++j) {
        if (src[j] != 255) {
          minRow = std::min(i, minRow);
          minCol = std::min(j, minCol);
          maxRow = std::max(i, maxRow);
          maxCol = std::max(j, maxCol);
        }
      }
    }

    cv::Mat trimmedScan(maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    for (int i = minRow; i < maxRow + 1; ++i) {
      const uchar *src = scan.ptr<uchar>(i);
      uchar *dst = trimmedScan.ptr<uchar>(i - minRow);
      for (int j = minCol; j < maxCol + 1; ++j) {
        dst[j - minCol] = src[j];
      }
    }

    trimmedScans.push_back(trimmedScan);
    zeroZero[k][0] -= minCol;
    zeroZero[k][1] -= minRow;
    ++k;
  }
}

void place::savePlacement(const std::vector<const place::posInfo *> &minima,
                          const std::string &outName,
                          const std::vector<Eigen::Vector2i> &zeroZero) {
  constexpr int maxToSave = 500;

  std::ofstream out(outName, std::ios::out);
  std::ofstream outB(outName.substr(0, outName.find(".")) + ".dat",
                     std::ios::out | std::ios::binary);

  const int num = minima.size();

  out << "Score x y rotation" << std::endl;
  outB.write(reinterpret_cast<const char *>(&num), sizeof(num));
  for (int i = 0; i < std::min(maxToSave, num); ++i) {
    place::posInfo minScore = *minima[i];
    minScore.x += zeroZero[minScore.rotation][0];
    minScore.y += zeroZero[minScore.rotation][1];
    out << minScore.score << " " << minScore.x << " " << minScore.y << " "
        << minScore.rotation << std::endl;

    outB.write(reinterpret_cast<const char *>(&minScore), sizeof(minScore));
  }
  out.close();
  outB.close();
}

bool place::reshowPlacement(const std::string &scanName,
                            const std::string &zerosFile,
                            const std::string &doorName,
                            const place::DoorDetector &d,
                            const std::string &preDone) {
  const std::string buildName = scanName.substr(scanName.find("_") - 3, 3);
  const std::string scanNumber = scanName.substr(scanName.find(".") - 3, 3);
  const std::string placementName =
      buildName + "_placement_" + scanNumber + ".dat";

  std::ifstream in(preDone + placementName, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return false;
  if (!FLAGS_reshow)
    return true;

  if (!FLAGS_quietMode)
    std::cout << placementName << std::endl;

  std::vector<cv::Mat> rotatedScans, toTrim;
  std::vector<Eigen::Vector2i> zeroZero;
  place::loadInScans(scanName, zerosFile, toTrim, zeroZero);
  place::trimScans(toTrim, rotatedScans, zeroZero);

  std::vector<std::vector<place::Door>> doors = loadInDoors(doorName, zeroZero);

  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));
  std::vector<place::posInfo> scores(num);
  for (auto &s : scores)
    in.read(reinterpret_cast<char *>(&s), sizeof(place::posInfo));

  int cutOffNum = place::getCutoffIndex(
      placementName, scores, [](const place::posInfo &s) { return s.score; });
  cutOffNum = FLAGS_top > 0 ? FLAGS_top : cutOffNum;

  num = std::min(num, cutOffNum);

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);

  if (!FLAGS_quietMode)
    std::cout << "Showing minima: " << num << std::endl;

  for (int k = 0; k < std::min(num, (int)scores.size());) {
    auto &currentScore = scores[k];

    const cv::Mat &bestScan = rotatedScans[currentScore.rotation];

    const int xOffset = currentScore.x - zeroZero[currentScore.rotation][0];
    const int yOffset = currentScore.y - zeroZero[currentScore.rotation][1];

    cv::Mat_<cv::Vec3b> output = fpColor.clone();

    auto &res = d.getResponse(0);
    for (int i = 0; i < res.outerSize(); ++i)
      for (Eigen::SparseMatrix<char>::InnerIterator it(res, i); it; ++it)
        if (it.value() > 1)
          output(it.row(), it.col()) = cv::Vec3b(0, 255, 0);

    cv::imwrite("door_labeled_floorplan.png", output);
    for (int j = 0; j < bestScan.rows; ++j) {
      if (j + yOffset < 0 || j + yOffset >= fpColor.rows)
        continue;
      const uchar *src = bestScan.ptr<uchar>(j);
      for (int i = 0; i < bestScan.cols; ++i) {
        if (i + xOffset < 0 || i + xOffset >= fpColor.cols)
          continue;

        if (src[i] != 255) {
          output(j + yOffset, i + xOffset) = cv::Vec3b(0, 0, 255 - src[i]);
        }
      }
    }

    for (int j = -10; j < 10; ++j)
      for (int i = -10; i < 10; ++i)
        output(j + currentScore.y, i + currentScore.x) = cv::Vec3b(255, 0, 0);

    for (auto &d : doors[currentScore.rotation]) {
      auto color = randomColor();
      for (double x = 0; x < d.w; ++x) {
        Eigen::Vector3i index =
            (d.corner + x * d.xAxis + Eigen::Vector3d(xOffset, yOffset, 0))
                .unaryExpr([](auto v) { return std::round(v); })
                .cast<int>();

        for (int k = -2; k <= 2; ++k) {
          for (int l = -2; l <= 2; ++l) {
            output(index[1] + k, index[0] + l) = color;
          }
        }
      }
    }

    if (!FLAGS_quietMode) {
      std::cout << &currentScore << std::endl;
      std::cout << "% of scan unexplained: "
                << currentScore.scanFP / currentScore.scanPixels
                << "   Index: " << k << std::endl
                << std::endl;
    }

    const int keyCode = cv::rectshow(output);

    if (keyCode == 27) {
      cv::imwrite(preDone + buildName + "_ss_" + scanNumber + ".png", output);
      break;
    } else if (keyCode == 8)
      k = k > 0 ? k - 1 : k;
    else
      ++k;
  }
  return true;
}

void place::displayOutput(
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const std::vector<const place::posInfo *> &minima) {

  if (!FLAGS_quietMode) {
    std::cout << "Num minima: " << minima.size() << std::endl;
    std::cout << "Press a key to begin displaying placement options"
              << std::endl;
  }

  cv::rectshow(fpColor);

  int currentCount = 0;
  for (auto &min : minima) {
    const int xOffset = min->x;
    const int yOffset = min->y;
    const Eigen::SparseMatrix<double> &currentScan =
        rSSparseTrimmed[min->rotation];
    cv::Mat output(fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
    fpColor.copyTo(output);

    cv::Mat_<cv::Vec3b> _output = output;

    for (int i = 0; i < currentScan.outerSize(); ++i) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it;
           ++it) {
        if (it.row() + yOffset < 0 || it.row() + yOffset >= output.rows)
          continue;
        if (it.col() + xOffset < 0 || it.col() + xOffset >= output.cols)
          continue;

        _output(it.row() + yOffset, it.col() + xOffset)[0] = 0;
        _output(it.row() + yOffset, it.col() + xOffset)[1] = 0;
        _output(it.row() + yOffset, it.col() + xOffset)[2] = 255;
      }
    }

    if (!FLAGS_quietMode) {
      std::cout << min << std::endl << std::endl;
    }
    const int keyCode = cv::rectshow(output);
    ~output;
    if (keyCode == 27)
      break;
  }
}

void place::displayOutput(
    const Eigen::SparseMatrix<double> &fp,
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const Eigen::MatrixXb &fpDoors,
    const std::vector<std::vector<place::Door>> &pcDoors,
    const std::vector<const place::posInfo *> &minima) {
  const int num = minima.size() < 20 ? minima.size() : 20;
  if (!FLAGS_quietMode) {
    std::cout << "Num minima: " << num << std::endl;
    std::cout << "Press a key to begin displaying placement options"
              << std::endl;
  }
  cv::Mat fpImg = place::sparseToImage(fp);
  cv::Mat tmpColor(fpImg.rows, fpImg.cols, CV_8UC3, cv::Scalar::all(255));

  for (int i = 0; i < tmpColor.rows; ++i) {
    uchar *dst = tmpColor.ptr<uchar>(i);
    const uchar *src = fpImg.ptr<uchar>(i);
    for (int j = 0; j < tmpColor.cols; ++j) {
      if (src[j] != 255) {
        dst[j * 3] = 128;
        dst[j * 3 + 1] = 128;
        dst[j * 3 + 2] = 128;
      }
    }
  }

  for (int j = 0; j < fpDoors.rows(); ++j) {
    uchar *dst = tmpColor.ptr<uchar>(j);
    for (int i = 0; i < fpDoors.cols(); ++i) {
      if (fpDoors(j, i)) {
        dst[i * 3] = 0;
        dst[i * 3 + 1] = 255;
        dst[i * 3 + 2] = 0;
      }
    }
  }

  cv::rectshow(tmpColor);
  const int cutOff = FLAGS_top > 0 ? FLAGS_top : 20;

  int currentCount = 0;
  for (auto &min : minima) {
    const int xOffset = min->x;
    const int yOffset = min->y;
    const Eigen::SparseMatrix<double> &currentScan =
        rSSparseTrimmed[min->rotation];
    auto &doors = pcDoors[min->rotation];

    cv::Mat output(tmpColor.rows, tmpColor.cols, CV_8UC3, cv::Scalar::all(255));
    tmpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;

    for (int i = 0; i < currentScan.outerSize(); ++i) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it;
           ++it) {
        if (it.row() + yOffset < 0 || it.row() + yOffset >= output.rows)
          continue;
        if (it.col() + xOffset < 0 || it.col() + xOffset >= output.cols)
          continue;

        _output(it.row() + yOffset, it.col() + xOffset)[0] = 0;
        _output(it.row() + yOffset, it.col() + xOffset)[1] = 0;
        _output(it.row() + yOffset, it.col() + xOffset)[2] = 255;
      }
    }

    for (auto &d : doors) {
      auto color = randomColor();
      for (double x = 0; x < d.w; ++x) {
        Eigen::Vector3i index =
            (d.corner + x * d.xAxis + Eigen::Vector3d(min->x, min->y, 0))
                .unaryExpr([](auto v) { return std::round(v); })
                .cast<int>();

        _output(index[1], index[0]) = color;
      }
    }

    if (!FLAGS_quietMode) {
      std::cout << min << std::endl << std::endl;
    }
    cv::rectshow(output);
    ~output;
    if (++currentCount == cutOff)
      break;
  }
}

void place::loadInTruePlacement(const std::string &scanName,
                                const std::vector<Eigen::Vector2i> &zeroZero) {
  const std::string placementName =
      FLAGS_outputV1 + scanName.substr(scanName.find("_") - 3, 3) +
      "_placement_" + scanName.substr(scanName.find(".") - 3, 3) + ".dat";
  std::ifstream in(placementName, std::ios::in | std::ios::binary);

  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));

  std::vector<place::posInfo> tmp(num);
  for (int i = 0; i < num; ++i) {
    in.read(reinterpret_cast<char *>(&tmp[i]), sizeof(place::posInfo));
  }

  truePlacement.clear();
  for (auto &s : tmp) {
    Eigen::Vector3i tmp2(s.x - zeroZero[s.rotation][0],
                         s.y - zeroZero[s.rotation][1], s.rotation);
    truePlacement.push_back(tmp2);
  }
}

void place::displayTruePlacement(
    const std::vector<Eigen::SparseMatrix<double>> &rSSparseTrimmed,
    const std::vector<place::posInfo> &scores,
    const std::vector<Eigen::Vector2i> &zeroZero) {

  std::vector<const place::posInfo *> tmp;
  for (int i = 0; i < scores.size(); ++i) {
    tmp.push_back(&scores[i]);
  }

  std::cout << "displaying true placement" << std::endl;
  place::displayOutput(rSSparseTrimmed, tmp);
}

void place::scanToSparse(const cv::Mat &scan,
                         Eigen::SparseMatrix<double> &sparse) {
  std::vector<Eigen::Triplet<double>> tripletList;

  for (int i = 0; i < scan.rows; ++i) {
    const uchar *src = scan.ptr<uchar>(i);
    for (int j = 0; j < scan.cols; ++j) {
      if (src[j] == 255)
        continue;
      double confidence = 1.0 - (double)src[j] / 255.0;
      tripletList.push_back(Eigen::Triplet<double>(i, j, confidence));
    }
  }
  sparse = Eigen::SparseMatrix<double>(scan.rows, scan.cols);
  sparse.setFromTriplets(tripletList.begin(), tripletList.end());
  sparse.makeCompressed();
  sparse.prune(1.0);
}

Eigen::SparseMatrix<double> place::scanToSparse(const cv::Mat &scan) {
  std::vector<Eigen::Triplet<double>> tripletList;

  for (int i = 0; i < scan.rows; ++i) {
    const uchar *src = scan.ptr<uchar>(i);
    for (int j = 0; j < scan.cols; ++j) {
      if (src[j] == 255)
        continue;
      double confidence = 1.0 - (double)src[j] / 255.0;
      tripletList.push_back(Eigen::Triplet<double>(i, j, confidence));
    }
  }
  Eigen::SparseMatrix<double> sparseTmp(scan.rows, scan.cols);
  sparseTmp.setFromTriplets(tripletList.begin(), tripletList.end());
  sparseTmp.makeCompressed();
  sparseTmp.prune(1.0);

  return sparseTmp;
}

void place::displayScanAndMask(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>
        &rSSparsePyramidTrimmed,
    const std::vector<std::vector<Eigen::MatrixXb>> &eMaskPyramidTrimmedNS) {

  for (int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
    for (int j = 0; j < rSSparsePyramidTrimmed[i].size(); ++j) {
      const Eigen::SparseMatrix<double> &currentScan =
          rSSparsePyramidTrimmed[i][j];
      const Eigen::MatrixXb &currentMask = eMaskPyramidTrimmedNS[i][j];
      cv::Mat out(currentScan.rows(), currentScan.cols(), CV_8UC3,
                  cv::Scalar::all(255));

      for (int i = 0; i < out.rows; ++i) {
        uchar *dst = out.ptr<uchar>(i);
        for (int j = 0; j < out.cols; ++j) {
          if (currentMask(i, j) != 0) {
            dst[3 * j] = 0;
            dst[3 * j + 1] = 0;
            dst[3 * j + 2] = 0;
          }
        }
      }

      cv::Mat_<cv::Vec3b> _out = out;
      for (int i = 0; i < currentScan.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it;
             ++it) {
          if (it.value() > 0 && _out(it.row(), it.col())[0] == 0) {
            _out(it.row(), it.col())[0] = 0;
            _out(it.row(), it.col())[1] = 255;
            _out(it.row(), it.col())[2] = 0;
          } else if (it.value() > 0) {
            _out(it.row(), it.col())[0] = 0;
            _out(it.row(), it.col())[1] = 0;
            _out(it.row(), it.col())[2] = 255;
          }
        }
      }
      out = _out;
      cv::rectshow(out);
    }
  }
}

void place::erodeSparse(const Eigen::SparseMatrix<double> &src,
                        Eigen::SparseMatrix<double> &dst) {
  dst = Eigen::SparseMatrix<double>(src.rows(), src.cols());
  std::vector<Eigen::Triplet<double>> tripletList;
  Eigen::MatrixXd srcNS = Eigen::MatrixXd(src);

  for (int i = 0; i < srcNS.cols(); ++i) {
    for (int j = 0; j < srcNS.rows(); ++j) {
      double value = 0.0;
      for (int k = -1; k < 1; ++k) {
        for (int l = -1; l < 1; ++l) {
          if (i + k < 0 || i + k >= srcNS.cols() || j + l < 0 ||
              j + l >= srcNS.rows())
            continue;
          else
            value = std::max(value, srcNS(j + l, i + k));
        }
      }

      if (value != 0)
        tripletList.push_back(Eigen::Triplet<double>(j, i, value));
    }
  }
  dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

static void labelNeighbours(const cv::Mat &image, const int currentLabel,
                            Eigen::RowMatrixXi &labeledImage,
                            std::list<std::pair<int, int>> &toLabel) {

  if (toLabel.empty())
    return;

  int yOffset, xOffset;
  std::tie(yOffset, xOffset) = toLabel.front();
  toLabel.pop_front();

  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      if (j + yOffset < 0 || j + yOffset >= labeledImage.rows())
        continue;
      if (i + xOffset < 0 || i + xOffset >= labeledImage.cols())
        continue;
      if (image.at<uchar>(j + yOffset, i + xOffset) != 255 &&
          labeledImage(j + yOffset, i + xOffset) == 0) {
        labeledImage(j + yOffset, i + xOffset) = currentLabel;
        toLabel.emplace_front(j + yOffset, i + xOffset);
      }
    }
  }

  labelNeighbours(image, currentLabel, labeledImage, toLabel);
}

void place::removeMinimumConnectedComponents(cv::Mat &image) {
  std::list<std::pair<int, int>> toLabel;
  Eigen::RowMatrixXi labeledImage =
      Eigen::RowMatrixXi::Zero(image.rows, image.cols);
  int currentLabel = 1;
  for (int j = 0; j < image.rows; ++j) {
    const uchar *src = image.ptr<uchar>(j);
    for (int i = 0; i < image.cols; ++i) {
      if (src[i] != 255 && labeledImage(j, i) == 0) {
        labeledImage(j, i) = currentLabel;
        toLabel.emplace_front(j, i);

        labelNeighbours(image, currentLabel, labeledImage, toLabel);

        ++currentLabel;
      }
    }
  }

  Eigen::VectorXi countPerLabel = Eigen::VectorXi::Zero(currentLabel);
  const int *labeledImagePtr = labeledImage.data();
  for (int i = 0; i < labeledImage.size(); ++i)
    ++countPerLabel[*(labeledImagePtr + i)];

  double average = 0.0, sigma = 0.0;
  const int *countPerLabelPtr = countPerLabel.data();
  std::tie(average, sigma) = place::aveAndStdev(
      countPerLabelPtr + 1, countPerLabelPtr + countPerLabel.size());

  double threshHold = average;
  for (int j = 0; j < image.rows; ++j) {
    uchar *src = image.ptr<uchar>(j);
    for (int i = 0; i < image.cols; ++i) {
      if (src[i] != 255) {
        const int label = labeledImage(j, i);
        const int count = countPerLabel[label];
        if (count < threshHold)
          src[i] = 255;
      }
    }
  }
}

std::vector<std::vector<place::Door>>
place::loadInDoors(const std::string &name,
                   const std::vector<Eigen::Vector2i> &zeroZero) {
  std::ifstream in(name, std::ios::in | std::ios::binary);
  std::vector<std::vector<place::Door>> tmp(NUM_ROTS);
  for (int r = 0; r < NUM_ROTS; ++r) {
    int num;
    in.read(reinterpret_cast<char *>(&num), sizeof(num));
    tmp[r].resize(num);
    for (auto &d : tmp[r]) {
      d.loadFromFile(in);
      d.corner *= buildingScale.getScale();
      d.corner += Eigen::Vector3d(zeroZero[r][0], zeroZero[r][1], 0);

      d.w *= buildingScale.getScale();
    }
  }
  in.close();

  return tmp;
}