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
static void selectR1Nodes(const std::vector<place::node> &nodes,
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
    const int index = std::min(5, static_cast<int>(numberOfLabels[i]));

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
  auto[average, sigma] = utils::ave_and_stdev(
      nodes, 0.0, [](const place::SelectedNode &n) { return n.agreement; },
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
    const fs::path folder = fs::path(FLAGS_voxelFolder) / "R0";

    utils::parse_folder(folder, pointVoxelFileNames, [](const fs::path &s) {
      return s.string().find("point") != std::string::npos;
    });
    for (auto &f : pointVoxelFileNames)
      f = f.filename();

    utils::parse_folder(folder, freeVoxelFileNames, [](const fs::path &s) {
      return s.string().find("freeSpace") != std::string::npos;
    });
    for (auto &f : freeVoxelFileNames)
      f = f.filename();

    std::sort(pointVoxelFileNames.begin(), pointVoxelFileNames.end());
    std::sort(freeVoxelFileNames.begin(), freeVoxelFileNames.end());
  }

  const fs::path metaDataFolder = fs::path(FLAGS_voxelFolder) / "metaData";
  utils::parse_folder(metaDataFolder, metaDataFiles);
  std::sort(metaDataFiles.begin(), metaDataFiles.end());

  const int numScans = pointFileNames.size();

  zeroZeros.resize(numScans);
  place::loadInScansGraph(pointFileNames, freeFileNames, zerosFileNames, scans,
                          masks, zeroZeros);

  for (int i = FLAGS_startIndex; i < std::min(FLAGS_startIndex + FLAGS_numScans,
                                              (int)pointFileNames.size());
       ++i) {
    const fs::path imageName = fs::path(FLAGS_dmFolder) / pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }

  R1Nodes.clear();
  selectR1Nodes(nodes, R1Nodes);
  calcNeighbors(R1Nodes, unwantedNeighbors);
}

void multi::Labeler::load() {
  static bool loaded = false;
  if (loaded)
    return;

  const fs::path rotFolder = FLAGS_rotFolder;
  utils::parse_folder(rotFolder, rotationsFiles);

  const fs::path panoImagesFolder = fs::path(FLAGS_panoFolder) / "images";
  utils::parse_folder(panoImagesFolder, panoFiles);

  std::vector<fs::path> panoDataNames;
  const fs::path panoDataFolder = fs::path(FLAGS_panoFolder) / "data";
  utils::parse_folder(panoDataFolder, panoDataNames);

  rotationMatricies.assign(rotationsFiles.size(),
                           std::vector<Eigen::Matrix3d>(NUM_ROTS));
  for (int j = 0; j < rotationsFiles.size(); ++j) {
    const fs::path &rotName = rotationsFiles[j];
    std::ifstream in = utils::open(rotName, std::ios::binary | std::ios::in);
    for (int i = 0; i < NUM_ROTS; ++i)
      in.read(reinterpret_cast<char *>(rotationMatricies[j][i].data()),
              sizeof(Eigen::Matrix3d));

    in.close();
  }
  panoramas.resize(panoFiles.size());
  for (int i = 0; i < panoFiles.size(); ++i) {
    const fs::path &imgName = panoFiles[i];
    const fs::path &dataName = panoDataNames[i];
    panoramas[i].loadFromFile(imgName, dataName);
  }

  voxelInfo.assign(metaDataFiles.size(),
                   std::vector<place::MetaData>(NUM_ROTS));
  for (int i = 0; i < metaDataFiles.size(); ++i) {
    const fs::path &metaName = metaDataFiles[i];
    std::ifstream in = utils::open(metaName, std::ios::in | std::ios::binary);
    for (int j = 0; j < NUM_ROTS; ++j) {
      voxelInfo[i][j].loadFromFile(in);
    }
    in.close();
  }

  loaded = true;
}

void multi::Labeler::weightEdges() {
  if (FLAGS_redo || !place::reloadGraph(adjacencyMatrix, 2)) {
    if (FLAGS_redo || !place::reloadGraph(adjacencyMatrix, 0)) {
      load();
      place::weightEdges(R1Nodes, voxelInfo, pointVoxelFileNames,
                         freeVoxelFileNames, rotationMatricies, panoramas,
                         adjacencyMatrix);
      if (FLAGS_save)
        place::saveGraph(adjacencyMatrix, 0);
    }

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

          const double weight = -0.25 * (weights[i] + weights[j]);
          adjacencyMatrix(id1, id2).hWeight += weight;
          adjacencyMatrix(id2, id1).hWeight += weight;
        }
      }
    }
    highOrderV2.clear();
    if (FLAGS_save)
      place::saveGraph(adjacencyMatrix, 1);
  }
  place::normalizeWeights(adjacencyMatrix, R1Nodes);
}

void multi::Labeler::solveTRW() {
  std::vector<place::R2Node> tmp;
  for (auto &n : R1Nodes)
    tmp.emplace_back(n, false);

  place::TRWSolver(adjacencyMatrix, R1Nodes, bestNodes);
  unlockNodes(bestNodes);
  exclusionLite(bestNodes, unwantedNeighbors);
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

  load();

  const double scale = buildingScale.getScale();
  const Eigen::Vector3d center =
      Eigen::Vector3d(bestNodes[0].x, bestNodes[0].y, 0) / scale +
      Eigen::Vector3d(0, 0, panoramas[bestNodes[0].color].floorCoord);

  const fs::path name = fs::path(FLAGS_outputV2) / "final_{}.dat"_format(index);
  std::ofstream out(name.string(), std::ios::out | std::ios::binary);
  const int num = bestNodes.size();
  out.write(reinterpret_cast<const char *>(&num), sizeof(num));
  Eigen::Matrix3d zeroMat = Eigen::Matrix3d::Zero();

  for (auto &n : bestNodes) {
    Eigen::Matrix3d &rotMat =
        n.agreement != -1000 ? rotationMatricies[n.color][n.rotation] : zeroMat;
    Eigen::Vector3d trans(n.x, n.y, 0);
    trans /= scale;
    trans[2] = panoramas[n.color].floorCoord;
    trans = trans - center;
    out.write(reinterpret_cast<const char *>(rotMat.data()),
              sizeof(double) * rotMat.size());
    out.write(reinterpret_cast<const char *>(trans.data()),
              sizeof(decltype(trans)::Scalar) * trans.size());
  }
  out.close();
}
