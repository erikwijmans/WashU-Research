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

	{
	  std::string folder = FLAGS_voxelFolder + "R" + std::to_string(0) + "/";
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
	
	{
		DIR *dir;
		struct dirent *ent;
		const std::string metaDataFolder = FLAGS_voxelFolder + "metaData/";
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

	for (int i = 0; i < numScans; ++i) {
	  const std::string metaName = metaDataFolder + metaDataFiles[i];
	  std::ifstream in (metaName, std::ios::in | std::ios::binary);
	  place::metaData tmp;
	  std::vector<place::metaData> tmpVec;
	  for (int i = 0; i < NUM_ROTS; ++i) {
	    in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
	    tmpVec.push_back(tmp);
	  }
	  voxelInfo.push_back(tmpVec);
	}

	zeroZeros.resize(numScans);
	place::loadInScansGraph(pointFileNames, freeFileNames,
	  zerosFileNames, scans, masks, zeroZeros);
	
	const int numToParse = 20;
	const int nodeStart = 0;
	for (int i = nodeStart; i < std::min(numToParse + nodeStart,
	  (int)pointFileNames.size()); ++i) {
	  const std::string imageName = FLAGS_dmFolder + pointFileNames[i];
	  place::loadInPlacementGraph(imageName, nodes, i);
	}


	const int numNodes = nodes.size();
	adjacencyMatrix = Eigen::MatrixXE (numNodes, numNodes);

	/*place::weightEdges(nodes, voxelInfo, 
	  pointVoxelFileNames, freeVoxelFileNames, adjacencyMatrix);

	const double endTime = omp_get_wtime();
	std::cout << "Time: " << endTime - startTime << std::endl;

	place::normalizeWeights(adjacencyMatrix, nodes);

	place::displayGraph(adjacencyMatrix, nodes, scans, zeroZeros);

	std::map<std::vector<int>, double> highOrder;
	place::createHigherOrderTerms(scans, zeroZeros, nodes, highOrder);

	// place::displayHighOrder(highOrder, nodes, scans, zeroZeros);

	bestNodes.clear();
	// place::MIPSolver(adjacencyMatrix, highOrder, nodes, bestNodes);
	place::TRWSolver(adjacencyMatrix, nodes, bestNodes);
	while (true) {
	  place::displayBest(bestNodes, scans, zeroZeros);
	}*/
}

void Labeler::weightEdges() {
	place::weightEdges(nodes, voxelInfo, 
		pointVoxelFileNames, freeVoxelFileNames, adjacencyMatrix);
	place::normalizeWeights(adjacencyMatrix, nodes);
}

void Labeler::solveTRW() {
	place::TRWSolver(adjacencyMatrix, nodes, bestNodes);
}

void Labeler::solveMIP() {
	place::createHigherOrderTerms(scans, zeroZeros, nodes, highOrder);
	place::MIPSolver(adjacencyMatrix, highOrder, nodes, bestNodes);
}

void Labeler::displaySolution() {
	place::displayBest(bestNodes, scans, zeroZeros);
}

void Labeler::displayGraph() {
	place::displayGraph(adjacencyMatrix, nodes, scans, zeroZeros);
}