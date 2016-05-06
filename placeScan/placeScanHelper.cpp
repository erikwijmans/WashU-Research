#include "placeScan_placeScanHelper.h"

#include <dirent.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <list>

cv::Mat fpColor, floorPlan;
std::vector<Eigen::Vector3i> truePlacement;
const double maxDelta = 0.10, maxTotal = 0.20;

static const int minWrite = 20;


void place::parseFolders(std::vector<std::string> & pointFileNames,
  std::vector<std::string> & zerosFileNames,
  std::vector<std::string> * freeFileNames){

  DIR *dir;
  struct dirent *ent;
  const std::string newDmFolder = FLAGS_dmFolder + "R0/";
  if ((dir = opendir (newDmFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != "."
        && fileName.find("point") != std::string::npos){
        pointFileNames.push_back(fileName);
      } else if (freeFileNames && fileName != ".." && fileName != "."
        && fileName.find("freeSpace") != std::string::npos) {
        freeFileNames->push_back(fileName);
      }
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }

  if ((dir = opendir (FLAGS_zerosFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != "." ){
        zerosFileNames.push_back(fileName);
      }
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }

  if (pointFileNames.size() != zerosFileNames.size()){
    perror("Not the same number of scans as zeros!");
    exit(1);
  }

  sort(pointFileNames.begin(), pointFileNames.end());
  sort(zerosFileNames.begin(), zerosFileNames.end());
  if (freeFileNames)
    sort(freeFileNames->begin(), freeFileNames->end());
}

void place::loadInScans(const std::string & scanName,
  const std::string & zerosFile, std::vector<cv::Mat> & rotatedScans,
  std::vector<Eigen::Vector2i> & zeroZero) {

  zeroZero.resize(NUM_ROTS);
  std::ifstream binaryReader (zerosFile, std::ios::in | std::ios::binary);
  for (int i = 0; i < NUM_ROTS; ++i) {
    binaryReader.read(reinterpret_cast<char *>(zeroZero[i].data()),
      sizeof(Eigen::Vector2i));
  }
  binaryReader.close();

  for (int i = 0; i < NUM_ROTS; ++i) {
    std::string fullScanName = FLAGS_dmFolder + "R" + std::to_string(i) + "/"
      + scanName;

    rotatedScans.push_back(cv::imread(fullScanName, 0));

    if (!rotatedScans[i].data){
      std::cout << "Error reading scan" << std::endl;
      exit(1);
    }
  }


  if (FLAGS_tinyPreviewIn || FLAGS_visulization) {
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", rotatedScans[0]);
    cv::waitKey(0);
  }
}

void place::loadInScans(const std::string & scanName,
   const std::string & zerosFile, std::vector<cv::Mat> & rotatedScans) {

  for (int i = 0; i < NUM_ROTS; ++i) {
    std::string fullScanName = FLAGS_dmFolder + "R" + std::to_string(i) + "/"
      + scanName;

    rotatedScans.push_back(cv::imread(fullScanName, 0));

    if (!rotatedScans[i].data){
      std::cout << "Error reading scan" << std::endl;
      exit(1);
    }
  }


  if (FLAGS_tinyPreviewIn || FLAGS_visulization) {
    cvNamedWindow("Preview", CV_WINDOW_NORMAL);
    cv::imshow("Preview", rotatedScans[0]);
    cv::waitKey(0);
  }
}

void place::loadInScansAndMasks(const std::string & scanName,
    const std::string & zerosFile,
    const std::string & maskName, std::vector<cv::Mat> & rotatedScans,
    std::vector<cv::Mat> & masks, std::vector<Eigen::Vector2i> & zeroZero) {

    place::loadInScans(scanName, zerosFile, rotatedScans, zeroZero);
    place::loadInScans(maskName, zerosFile, masks);
}

void place::trimScans(const std::vector<cv::Mat> & toTrim,
  std::vector<cv::Mat> & trimmedScans, std::vector<Eigen::Vector2i> & zeroZero) {
  int k = 0;
  for (auto & scan : toTrim){
    int minRow = scan.rows;
    int minCol = scan.cols;
    int maxRow = 0;
    int maxCol = 0;

    for (int i = 0; i < scan.rows; ++i) {
      const uchar * src = scan.ptr<uchar>(i);
      for (int j = 0; j < scan.cols; ++j) {
        if (src[j] != 255) {
          minRow = std::min(i, minRow);
          minCol = std::min(j, minCol);
          maxRow = std::max(i, maxRow);
          maxCol = std::max(j, maxCol);
        }
      }
    }

    cv::Mat trimmedScan (maxRow - minRow + 1, maxCol - minCol + 1, CV_8UC1);
    for (int i = minRow; i < maxRow + 1; ++i) {
      const uchar * src = scan.ptr<uchar>(i);
      uchar * dst = trimmedScan.ptr<uchar>(i-minRow);
      for (int j = minCol; j < maxCol + 1; ++j) {
        dst[j-minCol] = src[j];
      }
    }

    trimmedScans.push_back(trimmedScan);
    zeroZero[k][0] -= minCol;
    zeroZero[k][1] -= minRow;
    ++k;
  }
}

static void normalize(const std::vector<const place::posInfo *> & minima,
  std::vector<place::posInfo> & out) {
  double average = 0;
  for (auto & m : minima)
    average += m->score;
  average /= minima.size();

  double sigma = 0;
  for (auto & m : minima)
    sigma += (m->score - average)*(m->score - average);
  sigma /= minima.size() - 1;
  sigma = sqrt(sigma);

  for (auto & m : minima) {
    place::posInfo minScore = *m;
    minScore.score = (minScore.score - average)/sigma;
    out.push_back(minScore);
  }
}

static int countNumToDeltas(const std::vector<const place::posInfo *> & minima) {
  int num = 0;
  double lastScore = 1.0;
  const double initailScore = minima[0]->score;
  for (auto & min : minima) {
    if (min->score - lastScore < maxDelta
      && min->score - initailScore < maxTotal)
      ++num;
    else
      break;
  }
  return num;
}

void place::savePlacement(const std::vector<const place::posInfo *> & minima,
  const std::string & outName, const std::vector<Eigen::Vector2i> & zeroZero){
  std::ofstream out (outName, std::ios::out);
  std::ofstream outB (outName.substr(0, outName.find(".")) + ".dat", std::ios::out | std::ios::binary);

  const int numToDeltas = countNumToDeltas(minima);
  int num = minWrite > numToDeltas ? minWrite : numToDeltas;
  num = num > minima.size() ? minima.size() : num;

  out << "Score x y rotation" << std::endl;
  outB.write(reinterpret_cast<const char *>(&num), sizeof(num));
  for (int i = 0; i < num; ++i) {
    place::posInfo minScore = *minima[i];
    minScore.x += zeroZero[minScore.rotation][0];
    minScore.y += zeroZero[minScore.rotation][1];
    out << minScore.score << " " << minScore.x  << " "
      << minScore.y << " " << minScore.rotation << std::endl;

    outB.write(reinterpret_cast<const char *> (&minScore), sizeof(minScore));
  }
  out.close();
  outB.close();
}

bool place::reshowPlacement(const std::string & scanName,
  const std::string & zerosFile, const std::string & preDone) {
  const std::string placementName = preDone + scanName.substr(scanName.find("_")-3, 3)
  + "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";

  std::ifstream in (placementName, std::ios::in | std::ios::binary);
  if (!in.is_open())
    return false;
  if (!FLAGS_reshow)
    return true;

  if (!FLAGS_quiteMode)
    std::cout << placementName << std::endl;

  std::vector<cv::Mat> rotatedScans, toTrim;
  std::vector<Eigen::Vector2i> zeroZero;
  place::loadInScans(scanName, zerosFile, toTrim, zeroZero);
  place::trimScans(toTrim, rotatedScans, zeroZero);

  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));
  num = FLAGS_top > 0 && num > FLAGS_top ? FLAGS_top : num;
  place::posInfo minScore;
  in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));
  in.seekg(sizeof(num));

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);

  const double initailScore = minScore.score;
  double lastScore = 1.0;
  if (!FLAGS_quiteMode)
    std::cout << "Showing minima: " << num << std::endl;

  for (int i = 0; i < num; ++i) {
    in.read(reinterpret_cast<char *>(&minScore), sizeof(minScore));

    const cv::Mat & bestScan = rotatedScans[minScore.rotation];

    const int xOffset = minScore.x - zeroZero[minScore.rotation][0];
    const int yOffset = minScore.y - zeroZero[minScore.rotation][1];

    cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3);
    fpColor.copyTo(output);

    for (int i = 0; i < bestScan.rows; ++i) {
      if (i + yOffset < 0 || i + yOffset >= fpColor.rows)
        continue;

      const uchar * src = bestScan.ptr<uchar>(i);
      uchar * dst = output.ptr<uchar>(i + yOffset);
      for (int j = 0; j < bestScan.cols; ++j) {
        if (j + xOffset < 0 || j + xOffset >= fpColor.cols)
          continue;

        if (src[j]!=255){
          const int x = 3*(j + xOffset);
          dst[x + 0] = 0;
          dst[x + 1] = 0;
          dst[x + 2] = 255 - src[j];
        }
      }
    }

    for (int i = -10; i < 10; ++i) {
      uchar * dst = output.ptr<uchar>(i + minScore.y);
      for (int j = -10; j < 10; ++j) {
        const int x = 3*(j + minScore.x);
        dst[x + 0] = 255;
        dst[x + 1] = 0;
        dst[x + 2] = 0;
      }
    }

    if (!FLAGS_quiteMode) {
      std::cout << &minScore << std::endl;
      std::cout << "% of scan unexplained: " << minScore.scanFP/minScore.scanPixels << std::endl << std::endl;
    }
    cv::imshow("Preview", output);
    cv::waitKey(0);

    if (minScore.score - initailScore > maxTotal) break;
    if (minScore.score - lastScore > maxDelta) break;
    lastScore = minScore.score;
  }
  return true;
}

void place::displayOutput(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed,
  const std::vector<const place::posInfo *> & minima) {

  const int numToDeltas = countNumToDeltas(minima);
  int num = minWrite > numToDeltas ? minWrite : numToDeltas;
  num = num > minima.size() ? minima.size() : num;
  if (!FLAGS_quiteMode) {
    std::cout << "Num minima: " << num << std::endl;
    std::cout << "Press a key to begin displaying placement options" << std::endl;
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", fpColor);
  cv::waitKey(0);
  const int cutOff = FLAGS_top > 0 ? FLAGS_top : num;

  int currentCount = 0;
  for (auto & min : minima){
    const int xOffset = min->x;
    const int yOffset = min->y;
    const Eigen::SparseMatrix<double> & currentScan =
      rSSparseTrimmed[min->rotation];
    cv::Mat output (fpColor.rows, fpColor.cols, CV_8UC3, cv::Scalar::all(255));
    fpColor.copyTo(output);

    cv::Mat_<cv::Vec3b> _output = output;

    for (int i = 0; i < currentScan.outerSize(); ++i) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
        if (it.row() + yOffset < 0 || it.row() + yOffset >= output.rows)
          continue;
        if (it.col() + xOffset < 0 || it.col() + xOffset >= output.cols)
          continue;

        _output(it.row() + yOffset, it.col() + xOffset)[0]=0;
        _output(it.row() + yOffset, it.col() + xOffset)[1]=0;
        _output(it.row() + yOffset, it.col() + xOffset)[2]=255;

      }
    }

    cv::imshow("Preview", output);
    if (!FLAGS_quiteMode) {
      std::cout << min << std::endl << std::endl;
    }
    cv::waitKey(0);
    ~output;
    if (++currentCount == cutOff) break;
  }
}

void place::displayOutput(const Eigen::SparseMatrix<double> & fp,
  const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed,
  const std::vector<const place::posInfo *> & minima) {
  const int num = minima.size() < 20 ? minima.size() : 20;
  if (!FLAGS_quiteMode) {
    std::cout << "Num minima: " << num << std::endl;
    std::cout << "Press a key to begin displaying placement options" << std::endl;
  }
  cv::Mat fpImg = place::sparseToImage(fp);
  cv::Mat tmpColor (fpImg.rows, fpImg.cols, CV_8UC3, cv::Scalar::all(255));

  for (int i = 0; i < tmpColor.rows; ++i) {
    uchar * dst = tmpColor.ptr<uchar>(i);
    const uchar * src = fpImg.ptr<uchar>(i);
    for (int j = 0; j < tmpColor.cols; ++j) {
      if (src[j]!=255) {
        dst[j*3] = 128;
        dst[j*3+1] = 128;
        dst[j*3+2] = 128;
      }
    }
  }

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);
  cv::imshow("Preview", tmpColor);
  cv::waitKey(0);
  const int cutOff = FLAGS_top > 0 ? FLAGS_top : 20;

  int currentCount = 0;
  for (auto & min : minima){
    const int xOffset = min->x;
    const int yOffset = min->y;
    const Eigen::SparseMatrix<double> & currentScan =
      rSSparseTrimmed[min->rotation];
    cv::Mat output (tmpColor.rows, tmpColor.cols, CV_8UC3, cv::Scalar::all(255));
    tmpColor.copyTo(output);
    cv::Mat_<cv::Vec3b> _output = output;

    for (int i = 0; i < currentScan.outerSize(); ++i) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(currentScan, i); it; ++it){
        if (it.row() + yOffset < 0 || it.row() + yOffset >= output.rows)
          continue;
        if (it.col() + xOffset < 0 || it.col() + xOffset >= output.cols)
          continue;

        _output(it.row() + yOffset, it.col() + xOffset)[0]=0;
        _output(it.row() + yOffset, it.col() + xOffset)[1]=0;
        _output(it.row() + yOffset, it.col() + xOffset)[2]=255;

      }
    }

    cv::imshow("Preview", output);
    if (!FLAGS_quiteMode) {
      std::cout << min << std::endl << std::endl;
    }
    cv::waitKey(0);
    ~output;
    if (++currentCount == cutOff) break;
  }
}

void place::loadInTruePlacement(const std::string & scanName,
  const std::vector<Eigen::Vector2i> & zeroZero){
  const std::string placementName = FLAGS_preDone + scanName.substr(scanName.find("_")-3, 3)
  + "_placement_" + scanName.substr(scanName.find(".")-3, 3) + ".dat";
  std::ifstream in (placementName, std::ios::in | std::ios::binary);

  int num;
  in.read(reinterpret_cast<char *>(&num), sizeof(num));

  std::vector<place::posInfo> tmp (num);
  for (int i = 0; i < num; ++i) {
    in.read(reinterpret_cast<char *>(&tmp[i]), sizeof(place::posInfo));
  }

  truePlacement.clear();
  for (auto & s : tmp){
    Eigen::Vector3i tmp2 (s.x - zeroZero[s.rotation][0],
      s.y - zeroZero[s.rotation][1], s.rotation);
    truePlacement.push_back(tmp2);
  }
}

void place::displayTruePlacement(const std::vector<Eigen::SparseMatrix<double> > & rSSparseTrimmed,
  const std::vector<place::posInfo> & scores,
  const std::vector<Eigen::Vector2i> & zeroZero){

  std::vector<const place::posInfo *> tmp;
  for (int i = 0; i < scores.size(); ++i) {
    tmp.push_back(&scores[i]);
  }

  std::cout << "displaying true placement" << std::endl;
  place::displayOutput(rSSparseTrimmed, tmp);
}

void place::sparseToImage(const Eigen::SparseMatrix<double> & toImage,
  cv::Mat & imageOut){

  imageOut = cv::Mat(toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));

  double maxV = 0;
  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
      maxV = std::max(maxV, it.value());
    }
  }

  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
      imageOut.at<uchar>(it.row(), it.col()) = 255 - 255*it.value()/maxV;
    }
  }
}

cv::Mat place::sparseToImage(const Eigen::SparseMatrix<double> & toImage){

  cv::Mat image (toImage.rows(), toImage.cols(), CV_8UC1, cv::Scalar::all(255));
  double maxV = 0;
  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
      maxV = std::max(maxV, it.value());
    }
  }

  for (int i = 0; i < toImage.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(toImage, i); it; ++it) {
      image.at<uchar>(it.row(), it.col()) = 255 - 255*it.value()/maxV;
    }
  }
  return image;
}

void place::scanToSparse(const cv::Mat & scan,
  Eigen::SparseMatrix<double> & sparse) {
  std::vector<Eigen::Triplet<double> > tripletList;

  for (int i = 0; i < scan.rows; ++i) {
    const uchar * src = scan.ptr<uchar>(i);
    for (int j = 0; j < scan.cols; ++j) {
      if (src[j] == 255)
        continue;
      double confidence = 1.0 -(double)src[j]/255.0;
      tripletList.push_back(Eigen::Triplet<double> (i,j,confidence));
    }
  }
  sparse = Eigen::SparseMatrix<double>(scan.rows, scan.cols);
  sparse.setFromTriplets(tripletList.begin(), tripletList.end());
  sparse.makeCompressed();
  sparse.prune(1.0);
}

Eigen::SparseMatrix<double> place::scanToSparse(const cv::Mat & scan) {
  std::vector<Eigen::Triplet<double> > tripletList;

  for (int i = 0; i < scan.rows; ++i) {
    const uchar * src = scan.ptr<uchar>(i);
    for (int j = 0; j < scan.cols; ++j) {
      if (src[j] == 255)
        continue;
      double confidence = 1.0 -(double)src[j]/255.0;
      tripletList.push_back(Eigen::Triplet<double> (i,j,confidence));
    }
  }
  Eigen::SparseMatrix<double> sparseTmp (scan.rows, scan.cols);
  sparseTmp.setFromTriplets(tripletList.begin(), tripletList.end());
  sparseTmp.makeCompressed();
  sparseTmp.prune(1.0);

  return sparseTmp;
}

void place::displayScanAndMask(const std::vector<std::vector<Eigen::SparseMatrix<double> > > & rSSparsePyramidTrimmed,
  const std::vector<std::vector<Eigen::MatrixXb> > & eMaskPyramidTrimmedNS) {

  for (int i = 0; i < rSSparsePyramidTrimmed.size(); ++i) {
    for (int j = 0; j < rSSparsePyramidTrimmed[i].size(); ++j) {
      const Eigen::SparseMatrix<double> & currentScan = rSSparsePyramidTrimmed[i][j];
      const Eigen::MatrixXb & currentMask = eMaskPyramidTrimmedNS[i][j];
      cv::Mat out (currentScan.rows(), currentScan.cols(), CV_8UC3, cv::Scalar::all(255));

      for (int i = 0; i < out.rows; ++i) {
        uchar * dst = out.ptr<uchar>(i);
        for (int j = 0; j < out.cols; ++j) {
          if (currentMask(i,j) != 0) {
            dst[3*j] = 0;
            dst[3*j+1] = 0;
            dst[3*j+2] = 0;
          }
        }
      }

      cv::Mat_<cv::Vec3b> _out = out;
      for (int i = 0; i < currentScan.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it (currentScan, i); it; ++it) {
          if (it.value() > 0 && _out(it.row(), it.col())[0] == 0) {
            _out(it.row(), it.col())[0] = 0;
            _out(it.row(), it.col())[1] = 255;
            _out(it.row(), it.col())[2] = 0;
          } else if ( it.value() > 0) {
            _out(it.row(), it.col())[0] = 0;
            _out(it.row(), it.col())[1] = 0;
            _out(it.row(), it.col())[2] = 255;
          }
        }
      }
      out = _out;
      cvNamedWindow("Preview", CV_WINDOW_NORMAL);
      cv::imshow("Preview", out);
      cv::waitKey(0);
    }
  }
}

void place::erodeSparse(const Eigen::SparseMatrix<double> & src,
  Eigen::SparseMatrix<double> & dst) {
  dst = Eigen::SparseMatrix<double>(src.rows(), src.cols());
  std::vector<Eigen::Triplet<double> > tripletList;
  Eigen::MatrixXd srcNS = Eigen::MatrixXd(src);

  for (int i = 0; i < srcNS.cols(); ++i) {
    for (int j = 0; j < srcNS.rows(); ++j) {
      double value = 0.0;
      for (int k = -1; k < 1; ++k) {
        for (int l = -1; l < 1; ++l) {
          if (i + k < 0 || i + k >=srcNS.cols() ||
            j+l < 0 || j + l >=srcNS.rows())
            continue;
          else
            value = std::max(value, srcNS(j + l, i + k));
        }
      }

      if (value != 0)
        tripletList.push_back(Eigen::Triplet<double> (j, i, value));
    }
  }
  dst.setFromTriplets(tripletList.begin(), tripletList.end());
}

static void labelNeighbours(const cv::Mat & image, const int currentLabel,
  Eigen::RowMatrixXi & labeledImage, std::list <std::pair<int, int> > & toLabel) {

  if (toLabel.empty()) return;

  auto currentPixel = toLabel.front();
  toLabel.pop_front();
  const int yOffset = currentPixel.first;
  const int xOffset = currentPixel.second;
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      if (j + yOffset < 0 || j + yOffset >= labeledImage.rows())
        continue;
      if (i + xOffset < 0 || i + xOffset >= labeledImage.cols())
        continue;
      if (image.at<uchar>(j + yOffset, i + xOffset) != 255
        && labeledImage(j + yOffset, i + xOffset) == 0) {
        labeledImage(j + yOffset, i + xOffset) = currentLabel;
        toLabel.emplace_front(j + yOffset, i + xOffset);
      }
    }
  }
  labelNeighbours(image, currentLabel, labeledImage, toLabel);
}

void place::removeMinimumConnectedComponents(cv::Mat & image) {
  std::list<std::pair<int, int> > toLabel;
  Eigen::RowMatrixXi labeledImage = Eigen::RowMatrixXi::Zero(image.rows, image.cols);
  int currentLabel = 1;
  for (int j = 0; j < image.rows; ++j) {
    const uchar * src = image.ptr<uchar>(j);
    for (int i = 0; i < image.cols; ++i) {
      if (src[i] != 255 && labeledImage(j,i) == 0) {
        labeledImage(j,i) = currentLabel;
        toLabel.emplace_front(j,i);

        labelNeighbours(image, currentLabel, labeledImage, toLabel);

        ++currentLabel;
      }
    }
  }

  Eigen::VectorXi countPerLabel = Eigen::VectorXi::Zero(currentLabel + 1);
  const int * labeledImagePtr = labeledImage.data();
  for (int i = 1; i < labeledImage.size(); ++i)
      ++countPerLabel[*(labeledImagePtr + i)];



  double average = 0.0, sigma = 0.0;
  const int * countPerLabelPtr = countPerLabel.data();
  for (int i = 1; i < countPerLabel.size(); ++i)
    average += *(countPerLabelPtr + i);

  average /= countPerLabel.size() - 1;

  for (int i = 1; i < countPerLabel.size(); ++i) {
    const int value = *(countPerLabelPtr + i);
    sigma += (value - average)*(value - average);
  }
  sigma /= countPerLabel.size() - 2;
  sigma = sqrt(sigma);

  double threshHold = average;
  for (int j = 0; j < image.rows; ++j) {
    uchar * src = image.ptr<uchar>(j);
    for (int i = 0; i < image.cols; ++i) {
      if (src[i] != 255) {
        const int label = labeledImage(j,i);
        const int count = countPerLabel[label];
        if (count < threshHold)
          src[i] = 255;
      }
    }
  }
}
