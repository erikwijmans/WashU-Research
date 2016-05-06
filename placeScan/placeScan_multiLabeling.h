#ifndef PLACE_SCAN_MULTI_LABELING_H
#define PLACE_SCAN_MULTI_LABELING_H

#include "placeScan_placeScan.h"
#include "placeScan_placeScanHelper.h"
#include "placeScan_placeScanHelper2.h"

namespace multi {

  class Labeler {
    private:
      Eigen::MatrixXE adjacencyMatrix;
      std::vector<place::node> nodes;
      std::vector<std::vector<Eigen::Vector2i> > zeroZeros;
      std::vector<const place::node *> bestNodes;
      std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames;
      std::vector<std::string > pointVoxelFileNames, freeVoxelFileNames;
      std::vector<std::string> metaDataFiles, rotationsFiles, panoFiles;
      std::vector<std::vector<place::metaData> > voxelInfo;
      std::vector<std::vector<Eigen::MatrixXb> > scans, masks;
      std::unordered_map<std::vector<int>, double> highOrder;
      std::vector<std::vector<Eigen::Matrix3d> > rotationMatricies;
      std::vector<place::Panorama> panoramas;
      std::vector<size_t> numberOfLabels;
      void loadInPanosAndRot();
    public:
      Labeler();
      void weightEdges();
      void solveTRW();
      void solveMIP();
      void displaySolution();
      void displayGraph();
      void getLabeling(std::vector<const place::node *> & labels);
  };

} // multi

#endif // PLACE_SCAN_MULTI_LABELING_H
