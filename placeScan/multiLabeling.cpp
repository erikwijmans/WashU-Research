#include "placeScan_multiLabeling.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

#include <math.h>
#include <dirent.h>
#include <omp.h>

#include <opencv2/core/eigen.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include "gurobi_c++.h"

multi::Labeler::Labeler() {

  place::parseFolders(pointFileNames,
    zerosFileNames, &freeFileNames);

  std::cout << "Starting up Labeler" << std::endl;

  {
    std::string folder = FLAGS_voxelFolder + "R0/";
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (folder.data())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string fileName = ent->d_name;
        if (fileName != ".." && fileName != "."
          && fileName.find("point") != std::string::npos){
          pointVoxelFileNames.push_back(fileName);
        } else if (fileName != ".." && fileName != "."
          && fileName.find("freeSpace") != std::string::npos) {
          freeVoxelFileNames.push_back(fileName);
        }
      }
      closedir (dir);
    }  else {
      /* could not open directory */
      perror ("");
      exit(-1);
    }
    std::sort(pointVoxelFileNames.begin(), pointVoxelFileNames.end());
    std::sort(freeVoxelFileNames.begin(), freeVoxelFileNames.end());
  }

  const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
  {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (metaDataFolder.data())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string fileName = ent->d_name;
        if (fileName != ".." && fileName != ".")
          metaDataFiles.push_back(fileName);
      }
      closedir (dir);
    }  else {
      /* could not open directory */
      perror ("");
      exit(-1);
    }
    std::sort(metaDataFiles.begin(), metaDataFiles.end());
  }


  const int numScans = pointFileNames.size();

  voxelInfo.assign(metaDataFiles.size(), std::vector<place::metaData> (NUM_ROTS));
  for (int i = 0; i < metaDataFiles.size(); ++i) {
    const std::string metaName = metaDataFolder + metaDataFiles[i];
    std::ifstream in (metaName, std::ios::in | std::ios::binary);

    for (int j = 0; j < NUM_ROTS; ++j) {
      voxelInfo[i][j].loadFromFile(in);
    }
    in.close();
  }

  {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (metaDataFolder.data())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string fileName = ent->d_name;
        if (fileName != ".." && fileName != ".")
          metaDataFiles.push_back(fileName);
      }
      closedir (dir);
    }  else {
      /* could not open directory */
      perror ("");
      exit(-1);
    }
    std::sort(metaDataFiles.begin(), metaDataFiles.end());
  }

  std::cout << "Done parsing folders" << std::endl;
  std::cout << "Loading in scans" << std::endl;
  zeroZeros.resize(numScans);
  place::loadInScansGraph(pointFileNames, freeFileNames,
    zerosFileNames, scans, masks, zeroZeros);

  std::cout << "Loading in panoramas and rot mats" << std::endl;
  loadInPanosAndRot();
  std::cout << "Loading in placements" << std::endl;
  const int numToParse = numScans;
  const int nodeStart = 0;
  for (int i = nodeStart; i < std::min(numToParse + nodeStart,
    (int)pointFileNames.size()); ++i) {
    const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
    place::loadInPlacementGraph(imageName, nodes, i);
  }

  const int numNodes = nodes.size();
  adjacencyMatrix = Eigen::MatrixXE (numNodes, numNodes);

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
  std::cout << "Leaving labeler init" << std::endl;
}

void multi::Labeler::loadInPanosAndRot() {

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (FLAGS_rotFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != ".")
        rotationsFiles.push_back(fileName);
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }

  std::string panoImagesFolder = FLAGS_panoFolder + "images/";
  std::string panoDataFolder = FLAGS_panoFolder + "data/";
  if ((dir = opendir (panoImagesFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != ".")
        panoFiles.push_back(fileName);
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }

  std::vector<std::string> panoDataNames;
  if ((dir = opendir (panoDataFolder.data())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string fileName = ent->d_name;
      if (fileName != ".." && fileName != ".")
        panoDataNames.push_back(fileName);
    }
    closedir (dir);
  }  else {
    /* could not open directory */
    perror ("");
    exit(-1);
  }

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
}

void multi::Labeler::weightEdges() {
  std::cout << "Weighting edges" << std::endl;
  const double startTime = omp_get_wtime();
  place::weightEdges(nodes, voxelInfo, pointVoxelFileNames,
    freeVoxelFileNames, rotationMatricies, panoramas, adjacencyMatrix);
  place::normalizeWeights(adjacencyMatrix, nodes);
  const double endTime = omp_get_wtime();
  const int totalTime = endTime - startTime;
  const int seconds = totalTime % 60;
  const int minutes = (totalTime % 3600)/60;
  const int hours = totalTime/3600;
  std::cout << "Time: " << hours << "h " << minutes << "m "
    << seconds << "s" << std::endl;
}

void multi::Labeler::solveTRW() {
  place::TRWSolver(adjacencyMatrix, nodes, bestNodes);
}

void multi::Labeler::solveMIP() {
  place::createHigherOrderTerms(scans, zeroZeros, nodes, highOrder);
  place::MIPSolver(adjacencyMatrix, highOrder, nodes, bestNodes);
}

void multi::Labeler::displaySolution() {
  place::displayBest(bestNodes, scans, zeroZeros);
}

void multi::Labeler::displayGraph() {
  place::displayGraph(adjacencyMatrix, nodes, scans, zeroZeros);
}