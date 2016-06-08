#include "placeScan_multiLabeling.h"
#include "highOrder.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

#include <math.h>
#include <dirent.h>

const int minScans = 20;
static constexpr int minNodes = 2;
static void selectR1Nodes(const std::vector<place::node> & nodes,
                          std::vector<place::node> & R1Nodes) {
  constexpr int maxR1 = 200;
  int currentColor = nodes[0].color;
  double lastScore = 1.0;
  double initailScore = nodes[0].s.score;
  int count = 0;
  for (auto & n : nodes) {
    if (n.color != currentColor) {
      currentColor = n.color;
      count = 0;
      lastScore = 1.0;
      initailScore = n.s.score;
    }
    if (++count > minNodes && ( count > maxR1
      || n.s.score - lastScore > maxDelta
      || n.s.score - initailScore > maxTotal)) continue;
    lastScore = n.s.score;
    R1Nodes.push_back(n);
    R1Nodes.back().pos = R1Nodes.size() - 1;
  }
}

static void selectR2Nodes(const std::vector<place::node> & nodes,
                            const std::vector<place::SelectedNode> & bestNodes,
                            std::vector<place::R2Node> & R2Nodes) {
  int currentColor = nodes[0].color;
  double lastScore = 1.0;
  double initailScore = nodes[0].s.score;
  int index = 0;
  for (auto & n : nodes) {
    while (bestNodes[index].color != n.color) {
      if (bestNodes[index].locked)
        R2Nodes.push_back(bestNodes[index]);

      ++index;
    }
    if (!bestNodes[index].locked) {
      if (n.color != currentColor) {
        currentColor = n.color;
        lastScore = 1.0;
        initailScore = n.s.score;
      }
      if (n.s.score - lastScore > maxDelta) continue;
      if (n.s.score - initailScore > maxTotal) continue;
      lastScore = n.s.score;
      R2Nodes.emplace_back(n, false);
    }
  }
}

static void exclusionLite(std::vector<place::SelectedNode> & nodes,
                           const std::string & buildName) {
  auto it = buildingToScale.find(buildName);
  if (it == buildingToScale.cend()) {
    std::cout << "Could not find building " << buildName << std::endl;
    exit(1);
  }
  const double scale = it->second;
  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = i + 1; j < nodes.size(); ++j) {
      const Eigen::Vector2d a (nodes[i].s.x, nodes[i].s.y);
      const Eigen::Vector2d b(nodes[j].s.x, nodes[j].s.y);
      const double dist = (a-b).norm()/scale;
      if (dist < 1.0) {
        nodes[i].locked = false;
        nodes[j].locked = false;
      }
    }
  }
}

static void unlockNodes(std::vector<place::SelectedNode> & nodes) {
  double average = 0;
  for (auto & n : nodes)
    average += n.agreement;
  average /= nodes.size();

  double sigma = 0;
  for (auto & n : nodes)
    sigma += (n.agreement - average)*(n.agreement - average);

  sigma /= nodes.size() - 1;
  sigma = sqrt(sigma);

  for (auto & n : nodes) {
    const double norm = (n.agreement - average)/sigma;
    if (norm < -1.0)
      n.locked = false;
  }
}

multi::Labeler::Labeler() {
  place::parseFolders(pointFileNames,
    zerosFileNames, &freeFileNames);

  std::cout << "Starting up Labeler" << std::endl;

  {
    std::string folder = FLAGS_voxelFolder + "R0/";
    for (auto & file : folderToIterator(folder)) {
      std::string fileName = file.path().filename().string();
      if (fileName != ".." && fileName != "."
        && fileName.find("point") != std::string::npos){
        pointVoxelFileNames.push_back(fileName);
      } else if (fileName != ".." && fileName != "."
        && fileName.find("freeSpace") != std::string::npos) {
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
  place::loadInScansGraph(pointFileNames, freeFileNames,
    zerosFileNames, scans, masks, zeroZeros);

  const int numToParse = numScans;
  const int nodeStart = 0;
  for (int i = nodeStart; i < std::min(numToParse + nodeStart,
    (int)pointFileNames.size()); ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }

  {
    size_t i = 0;
    const place::node * prevNode = &nodes[0];
    for (auto & n : nodes) {
      if (n.color == prevNode->color) {
        prevNode = &n;
        ++i;
      } else {
        numberOfLabels.push_back(i);
        i = 1;
        prevNode = &n;
      }
    }
    numberOfLabels.push_back(i);
  }
  R1Nodes.clear();
  selectR1Nodes(nodes, R1Nodes);
}

void multi::Labeler::load() {
  if (loaded) return;

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

  rotationMatricies.assign(rotationsFiles.size(), std::vector<Eigen::Matrix3d> (NUM_ROTS));
  for (int j = 0; j < rotationsFiles.size(); ++j) {
    const std::string rotName = FLAGS_rotFolder + rotationsFiles[j];
    std::ifstream in (rotName, std::ios::binary | std::ios::in);
    for (int i = 0; i < NUM_ROTS; ++i)
      in.read(reinterpret_cast<char *>(rotationMatricies[j][i].data()), sizeof(Eigen::Matrix3d));

  in.close();
  }
  panoramas.resize(panoFiles.size());
  for (int i = 0; i < panoFiles.size(); ++i) {
    const std::string imgName = panoImagesFolder + panoFiles[i];
    const std::string dataName = panoDataFolder + panoDataNames[i];
    panoramas[i].loadFromFile(imgName, dataName);
  }

  const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
  voxelInfo.assign(metaDataFiles.size(), std::vector<place::MetaData> (NUM_ROTS));
  for (int i = 0; i < metaDataFiles.size(); ++i) {
    const std::string metaName = metaDataFolder + metaDataFiles[i];
    std::ifstream in (metaName, std::ios::in | std::ios::binary);
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
      freeVoxelFileNames, rotationMatricies, panoramas, adjacencyMatrix);
    if (FLAGS_save)
      place::saveGraph(adjacencyMatrix, 0);
  }
  place::normalizeWeights(adjacencyMatrix, R1Nodes);
}

void multi::Labeler::solveTRW() {
  place::TRWSolver(adjacencyMatrix, R1Nodes, bestNodes);
  unlockNodes(bestNodes);
  exclusionLite(bestNodes, freeVoxelFileNames[0].substr(0, 3));
}

void multi::Labeler::solveMIP() {
  std::vector<place::R2Node> R2Nodes;
  selectR2Nodes(R1Nodes, bestNodes, R2Nodes);

  place::createHigherOrderTerms(scans, zeroZeros, R2Nodes, highOrder);
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