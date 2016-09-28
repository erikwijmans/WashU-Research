#include "highOrder.h"
#include "placeScan_multiLabeling.h"
#include "placeScan_placeScanHelper.h"

#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dirent.h>
#include <math.h>

const int minScans = 20;
static constexpr int minNodes = 2;
static void selectR1Nodes(const std::vector<std::string> &names,
                          const std::vector<place::node> &nodes,
                          std::vector<place::node> &R1Nodes) {
  std::vector<size_t> numberOfLabels;
  {
    size_t i = 0;
    const place::node *prevNode = &nodes[0];
    for (auto &n : nodes) {
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

  for (int i = 0, offset = 0; i < numberOfLabels.size();
       offset += numberOfLabels[i], ++i) {
    const std::string &imageName = names[i];
    const std::string placementName =
        imageName.substr(imageName.find("_") - 3, 3) + "_placement_" +
        imageName.substr(imageName.find(".") - 3, 3) + ".dat";
    int index = place::getCutoffIndex(
        placementName,
        std::vector<place::node>(nodes.begin() + offset,
                                 nodes.begin() + offset + numberOfLabels[i]),
        [](const place::node &n) { return n.score; });

    R1Nodes.insert(R1Nodes.end(), nodes.begin() + offset,
                   nodes.begin() + offset + index);
  }

  for (int i = 0; i < R1Nodes.size(); ++i)
    R1Nodes[i].id = i;
}

static void selectR2Nodes(const std::vector<place::node> &nodes,
                          const std::vector<place::SelectedNode> &bestNodes,
                          std::vector<place::R2Node> &R2Nodes) {
  int currentColor = -1;
  for (auto &n : nodes) {
    if (n.color != currentColor) {
      currentColor = n.color;
      if (bestNodes[currentColor].locked)
        R2Nodes.push_back(bestNodes[currentColor]);
    }
    if (!bestNodes[currentColor].locked) {
      R2Nodes.emplace_back(n, false);
    }
  }
}

static void calcNeighbors(
    const std::vector<place::node> &nodes,
    std::unordered_map<int, std::unordered_set<int>> &unwantedNeighbors) {
  const double scale = buildingScale.getScale();

  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = 0; j < nodes.size(); ++j) {
      if (i == j)
        continue;

      const Eigen::Vector2d a(nodes[i].x, nodes[i].y);
      const Eigen::Vector2d b(nodes[j].x, nodes[j].y);

      const double dist = (a - b).norm() / scale;

      if (dist < 1.5) {
        const int idA = nodes[i].id;
        const int idB = nodes[j].id;
        auto it = unwantedNeighbors.find(idA);
        if (it == unwantedNeighbors.cend())
          unwantedNeighbors.emplace(idA, std::unordered_set<int>({idB}));
        else
          it->second.emplace(idB);
      }
    }
  }
}

static void exclusionLite(
    std::vector<place::SelectedNode> &nodes,
    const std::unordered_map<int, std::unordered_set<int>> &unwantedNeighbors) {

  for (int i = 0; i < nodes.size(); ++i) {
    const int idA = nodes[i].id;
    auto it = unwantedNeighbors.find(idA);

    if (it == unwantedNeighbors.cend())
      continue;

    auto &n = it->second;

    for (int j = 0; j < nodes.size(); ++j) {
      const int idB = nodes[j].id;
      auto isNeighbor = n.find(idB);

      if (isNeighbor != n.cend() && nodes[i].locked && nodes[j].locked) {
        if (nodes[i].agreement < nodes[j].agreement)
          nodes[i].locked = false;
        else
          nodes[j].locked = false;
      }
    }
  }
}

static void unlockNodes(std::vector<place::SelectedNode> &nodes) {

  double average, sigma;
  std::tie(average, sigma) = place::aveAndStdev(
      nodes.begin(), nodes.end(),
      [](const place::SelectedNode &n) { return n.agreement; },
      [](const place::SelectedNode &n) {
        return n.locked && n.numberOfCandidates > 1;
      });

  std::cout << average << "  " << sigma << std::endl;
  for (auto &n : nodes) {
    if (n.numberOfCandidates > 1) {
      const double norm = (n.agreement - average) / sigma;
      n.norm = norm;
      if (norm < -0.75)
        n.locked = false;
    }
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
  selectR1Nodes(pointFileNames, nodes, R1Nodes);
  calcNeighbors(R1Nodes, unwantedNeighbors);
}

void multi::Labeler::load() {
  static bool loaded = false;
  if (loaded)
    return;

  const std::string rotFolder = FLAGS_rotFolder;
  parseFolder(rotFolder, rotationsFiles);

  std::string panoImagesFolder = FLAGS_panoFolder + "images/";
  parseFolder(panoImagesFolder, panoFiles);

  std::vector<std::string> panoDataNames;
  std::string panoDataFolder = FLAGS_panoFolder + "data/";
  parseFolder(panoDataFolder, panoDataNames);

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

  place::createHigherOrderTermsV2(masks, zeroZeros, R1Nodes, highOrderV2);

  for (auto &pair : highOrderV2) {
    auto &incident = pair.first;
    auto &weights = pair.second;
    for (int i = 0; i < incident.size(); ++i) {
      // R1Nodes[incident[i]].hWeight += weights[i];

      for (int j = 0; j < incident.size(); ++j) {
        const int id1 = incident[i];
        const int id2 = incident[j];

        if (R1Nodes[id1].color == R1Nodes[id2].color)
          continue;

        const double weight = -0.5 * std::max(weights[i], weights[j]);
        adjacencyMatrix(id1, id2).hWeight += weight;
        adjacencyMatrix(id2, id1).hWeight += weight;
      }
    }
  }
  highOrderV2.clear();
}

void multi::Labeler::solveTRW() {
  std::vector<place::R2Node> tmp;
  for (auto &n : R1Nodes)
    tmp.emplace_back(n, false);

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
  place::displayBest(bestNodes, scans, masks, zeroZeros);
}

void multi::Labeler::displayGraph() {
  place::displayGraph(adjacencyMatrix, R1Nodes, scans, zeroZeros);
}

void multi::Labeler::saveFinal(int index) {
  if (!FLAGS_save)
    return;

  const double scale = buildingScale.getScale();
  const Eigen::Vector3d center =
      Eigen::Vector3d(bestNodes[0].x, bestNodes[0].y, 0) / scale;

  std::ofstream out(FLAGS_outputV2 + "final_" + std::to_string(index) + ".dat",
                    std::ios::out | std::ios::binary);
  const int num = bestNodes.size();
  out.write(reinterpret_cast<const char *>(&num), sizeof(num));
  Eigen::Matrix3d zeroMat = Eigen::Matrix3d::Zero();
  load();
  for (auto &n : bestNodes) {
    Eigen::Matrix3d &rotMat =
        n.agreement != -1000 ? rotationMatricies[n.color][n.rotation] : zeroMat;
    Eigen::Vector3d trans(n.x, n.y, 0);
    trans /= scale;
    trans = trans - center;
    out.write(reinterpret_cast<const char *>(rotMat.data()),
              sizeof(Eigen::Matrix3d));
    out.write(reinterpret_cast<const char *>(trans.data()),
              sizeof(Eigen::Vector3d));
  }
  out.close();
}
