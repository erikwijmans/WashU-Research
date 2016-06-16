#include "highOrder.h"
#include "placeScan_multiLabeling.h"

#include <boost/math/distributions/students_t.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dirent.h>
#include <math.h>

const int minScans = 20;
static constexpr int minNodes = 2;
static void selectR1Nodes(const std::vector<place::node> &nodes,
                          std::vector<place::node> &R1Nodes) {
  constexpr int maxR1Nodes = 50;
  int currentColor = nodes[0].color;
  double lastScore = 1.0;
  double initailScore = nodes[0].s.score;
  int count = 0;
  for (auto &n : nodes) {
    if (n.color != currentColor) {
      currentColor = n.color;
      count = 0;
      lastScore = 1.0;
      initailScore = n.s.score;
    }
    if (++count > minNodes &&
        (count > maxR1Nodes || n.s.score - lastScore > maxDelta ||
         n.s.score - initailScore > maxTotal))
      continue;
    lastScore = n.s.score;
    R1Nodes.push_back(n);
    R1Nodes.back().pos = R1Nodes.size() - 1;
  }
}

static void selectR2Nodes(const std::vector<place::node> &nodes,
                          const std::vector<place::SelectedNode> &bestNodes,
                          std::vector<place::R2Node> &R2Nodes) {
  constexpr int maxR2Nodes = 50;
  int currentColor = -1;
  double lastScore = 1.0;
  double initailScore = nodes[0].s.score;
  int count = 0;
  for (auto &n : nodes) {
    if (n.color != currentColor) {
      currentColor = n.color;
      lastScore = 1.0;
      initailScore = n.s.score;
      count = 0;
      if (bestNodes[currentColor].locked)
        R2Nodes.push_back(bestNodes[currentColor]);
    }
    if (!bestNodes[currentColor].locked) {
      if (++count > minNodes &&
          (n.s.score - lastScore > maxDelta ||
           n.s.score - initailScore > maxTotal || count > maxR2Nodes))
        continue;
      lastScore = n.s.score;
      R2Nodes.emplace_back(n, false);
    }
  }
}

static void exclusionLite(
    std::vector<place::SelectedNode> &nodes,
    std::unordered_map<int, std::unordered_set<int>> &unwantedNeighbors) {
  const double scale = buildingScale.getScale();
  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = 0; j < nodes.size(); ++j) {
      if (i == j)
        continue;
      const Eigen::Vector2d a(nodes[i].s.x, nodes[i].s.y);
      const Eigen::Vector2d b(nodes[j].s.x, nodes[j].s.y);
      const double dist = (a - b).norm() / scale;
      if (dist < 1.5) {
        auto it = unwantedNeighbors.find(i);
        if (it == unwantedNeighbors.cend())
          unwantedNeighbors.emplace(i, std::unordered_set<int>({j}));
        else
          it->second.emplace(j);
        if (nodes[i].locked && nodes[j].locked) {
          nodes[i].locked = false;
          nodes[j].locked = false;
        }
      }
    }
  }
}

static void unlockNodes(std::vector<place::SelectedNode> &nodes) {
  double average = 0;
  int count = 0;
  for (auto &n : nodes) {
    if (n.locked) {
      average += n.agreement;
      ++count;
    }
  }
  average /= count;

  double sigma = 0;
  for (auto &n : nodes)
    if (n.locked)
      sigma += (n.agreement - average) * (n.agreement - average);

  sigma /= count - 1;
  sigma = sqrt(sigma);

  std::cout << average << "  " << sigma << std::endl;
  for (auto &n : nodes) {
    const double norm = (n.agreement - average) / sigma;
    n.norm = norm;
    if (norm < -0.75)
      n.locked = false;
  }
}

multi::Labeler::Labeler() {
  place::parseFolders(pointFileNames, zerosFileNames, &freeFileNames);

  std::cout << "Starting up Labeler" << std::endl;

  {
    std::string folder = FLAGS_voxelFolder + "R0/";
    for (auto &file : folderToIterator(folder)) {
      std::string fileName = file.path().filename().string();
      if (fileName != ".." && fileName != "." &&
          fileName.find("point") != std::string::npos) {
        pointVoxelFileNames.push_back(fileName);
      } else if (fileName != ".." && fileName != "." &&
                 fileName.find("freeSpace") != std::string::npos) {
        freeVoxelFileNames.push_back(fileName);
      }
    }

    std::sort(pointVoxelFileNames.begin(), pointVoxelFileNames.end());
    std::sort(freeVoxelFileNames.begin(), freeVoxelFileNames.end());
  }

  const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
  parseFolder(metaDataFolder, metaDataFiles);
  std::sort(metaDataFiles.begin(), metaDataFiles.end());

  const int numScans = pointFileNames.size();

  zeroZeros.resize(numScans);
  place::loadInScansGraph(pointFileNames, freeFileNames, zerosFileNames, scans,
                          masks, zeroZeros);

  for (int i = FLAGS_startIndex; i < std::min(FLAGS_startIndex + FLAGS_numScans,
                                              (int)pointFileNames.size());
       ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }

  R1Nodes.clear();
  selectR1Nodes(nodes, R1Nodes);
}

void multi::Labeler::load() {
  if (loaded)
    return;

  const std::string rotFolder = FLAGS_rotFolder;
  parseFolder(rotFolder, rotationsFiles);

  std::string panoImagesFolder = FLAGS_panoFolder + "images/";
  parseFolder(panoImagesFolder, panoFiles);

  std::vector<std::string> panoDataNames;
  std::string panoDataFolder = FLAGS_panoFolder + "data/";
  parseFolder(panoDataFolder, panoDataNames);

  std::sort(rotationsFiles.begin(), rotationsFiles.end());
  std::sort(panoFiles.begin(), panoFiles.end());
  std::sort(panoDataNames.begin(), panoDataNames.end());

  rotationMatricies.assign(rotationsFiles.size(),
                           std::vector<Eigen::Matrix3d>(NUM_ROTS));
  for (int j = 0; j < rotationsFiles.size(); ++j) {
    const std::string rotName = FLAGS_rotFolder + rotationsFiles[j];
    std::ifstream in(rotName, std::ios::binary | std::ios::in);
    for (int i = 0; i < NUM_ROTS; ++i)
      in.read(reinterpret_cast<char *>(rotationMatricies[j][i].data()),
              sizeof(Eigen::Matrix3d));

    in.close();
  }
  panoramas.resize(panoFiles.size());
  for (int i = 0; i < panoFiles.size(); ++i) {
    const std::string imgName = panoImagesFolder + panoFiles[i];
    const std::string dataName = panoDataFolder + panoDataNames[i];
    panoramas[i].loadFromFile(imgName, dataName);
  }

  const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
  voxelInfo.assign(metaDataFiles.size(),
                   std::vector<place::MetaData>(NUM_ROTS));
  for (int i = 0; i < metaDataFiles.size(); ++i) {
    const std::string metaName = metaDataFolder + metaDataFiles[i];
    std::ifstream in(metaName, std::ios::in | std::ios::binary);
    for (int j = 0; j < NUM_ROTS; ++j) {
      voxelInfo[i][j].loadFromFile(in);
    }
    in.close();
  }

  loaded = true;
}

void multi::Labeler::weightEdges() {
  if (FLAGS_redo || !place::reloadGraph(adjacencyMatrix, 0)) {
    load();
    place::weightEdges(R1Nodes, voxelInfo, pointVoxelFileNames,
                       freeVoxelFileNames, rotationMatricies, panoramas,
                       adjacencyMatrix);
    if (FLAGS_save)
      place::saveGraph(adjacencyMatrix, 0);
  }
  place::normalizeWeights(adjacencyMatrix, R1Nodes);
}

void multi::Labeler::solveTRW() {
  place::TRWSolver(adjacencyMatrix, R1Nodes, bestNodes);
  unlockNodes(bestNodes);
  exclusionLite(bestNodes, unwantedNeighbors);
}

void multi::Labeler::solveMIP() {
  std::vector<place::R2Node> R2Nodes;
  selectR2Nodes(R1Nodes, bestNodes, R2Nodes);

  place::createHigherOrderTerms(scans, zeroZeros, R2Nodes, unwantedNeighbors,
                                highOrder);
  // place::displayHighOrder(highOrder, R2Nodes, scans, zeroZeros);
  bestNodes.clear();
  place::MIPSolver(adjacencyMatrix, highOrder, R2Nodes, bestNodes);
}

void multi::Labeler::displaySolution() {
  place::displayBest(bestNodes, scans, zeroZeros);
}

void multi::Labeler::displayGraph() {
  place::displayGraph(adjacencyMatrix, R1Nodes, scans, zeroZeros);
}

void multi::Labeler::saveFinal(int index) {
  if (!FLAGS_save)
    return;

  const double scale = buildingScale.getScale();
  const Eigen::Vector3d center =
      Eigen::Vector3d(bestNodes[0].s.x, bestNodes[0].s.y, 0) / scale;

  std::ofstream out(FLAGS_outputV2 + "final_" + std::to_string(index) + ".dat",
                    std::ios::out | std::ios::binary);
  const int num = bestNodes.size();
  out.write(reinterpret_cast<const char *>(&num), sizeof(num));

  load();
  for (auto &n : bestNodes) {
    Eigen::Matrix3d &rotMat = rotationMatricies[n.color][n.s.rotation];
    Eigen::Vector3d trans(n.s.x, n.s.y, 0);
    trans /= scale;
    trans = trans - center;
    out.write(reinterpret_cast<const char *>(rotMat.data()),
              sizeof(Eigen::Matrix3d));
    out.write(reinterpret_cast<const char *>(trans.data()),
              sizeof(Eigen::Vector3d));
  }
  out.close();
}
