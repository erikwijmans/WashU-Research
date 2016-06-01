#pragma once
#ifndef PLACE_SCAN_MULTI_LABELING_H
#define PLACE_SCAN_MULTI_LABELING_H

#include "placeScan_placeScan.h"
#include "placeScan_placeScanHelper.h"
#include "placeScan_placeScanHelper2.h"

extern const int minScans;

namespace multi {

  class Labeler {
    public:
      Labeler();
      void weightEdges();
      void solveTRW();
      void solveMIP();
      void displaySolution();
      void displayGraph();
      void getLabeling(std::vector<const place::node *> & labels);

      typedef struct s{
        double w;
        int c;
        s (double w, int c) : w {w}, c {c} {};
      } s;
      typedef std::unordered_map<std::vector<int>, s> map;
    private:
      Eigen::MatrixXE adjacencyMatrix;
      std::vector<place::node> nodes, R1Nodes;
      std::vector<std::vector<Eigen::Vector2i> > zeroZeros;
      std::vector<place::SelectedNode> bestNodes;
      std::vector<std::string> pointFileNames, zerosFileNames, freeFileNames;
      std::vector<std::string > pointVoxelFileNames, freeVoxelFileNames;
      std::vector<std::string> metaDataFiles, rotationsFiles, panoFiles;
      std::vector<std::vector<place::MetaData> > voxelInfo;
      std::vector<std::vector<Eigen::MatrixXb> > scans, masks;
      map highOrder;
      std::vector<std::vector<Eigen::Matrix3d> > rotationMatricies;
      std::vector<place::Panorama> panoramas;
      std::vector<size_t> numberOfLabels;
      void load();
      bool loaded = false;
  };

} // multi

#endif // PLACE_SCAN_MULTI_LABELING_H
