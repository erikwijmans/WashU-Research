#include "scan_gflags.h"

#include <iostream>

DEFINE_bool(
    visulization, false,
    "Turns on all visualization options that do not impact performance");
DEFINE_bool(previewIn, false,
            "Turns on a preview of the scan before it is placed");
DEFINE_bool(previewOut, false,
            "Shows a preview of the scans placement before saving");
DEFINE_bool(redo, false,
            "Forces the program to redo the placement of all scans given");
DEFINE_bool(quietMode, true, "Very minimal status reports");
DEFINE_bool(tinyPreviewIn, false, "Shows the input scan before any processing");
DEFINE_bool(save, true, "Saves the placements to file");
DEFINE_bool(debugMode, false,
            "Turns save off, turns replace on, and forces the program to "
            "display the correct placement according to the placement files "
            "specified by the preDone flag");
DEFINE_bool(reshow, true, "Reshows the placement from a previous run");
DEFINE_bool(V1, false, "Only will use V1 placement");
DEFINE_bool(V2, false, "Ony will use V2 placement");
DEFINE_bool(pe, false, "Tells the program to only examine point evidence");
DEFINE_bool(fe, false, "Tells the program to only examine free space evidence");
DEFINE_bool(preview, false, "Turns on previews of the output");
DEFINE_bool(3D, false, "writes out 3D voxelGrids");
DEFINE_bool(2D, false, "Creates 2D density maps");
DEFINE_string(floorPlan, "floorPlan.png",
              "Path to the floor plan that the scan should be placed on.  This "
              "will be appended to the dataPath.");
DEFINE_string(dmFolder, "densityMaps/",
              "Path to folder containing densityMaps. This will be appended to "
              "the dataPath.");
DEFINE_string(outputV1, "placementOptions/V1/",
              "Path to folder where the output of V1 will be written. This "
              "will be appended to the dataPath.");
DEFINE_string(outputV2, "placementOptions/V2/",
              "Path to folder where the output of V2 will be put. This will be "
              "appended to the dataPath.");
DEFINE_string(zerosFolder, "densityMaps/zeros/",
              "Path to folder where the pixel coordinates of (0,0) are. This "
              "will be appended to the dataPath.");
DEFINE_string(voxelFolder, "voxelGrids/",
              "Path to the folder where the voxelGrids are saved to. This will "
              "be appended to the dataPath.");
DEFINE_string(rotFolder, "densityMaps/rotations/",
              "Path from dataPath to rotation files");
DEFINE_string(panoFolder, "panoramas/", "path from dataPath to panoramas");
DEFINE_string(PTXFolder, "PTXFiles/", "Path to PTXFiles");
DEFINE_string(normalsFolder, "cloudNormals/", "name of normalsFolder");
DEFINE_string(descriptorsFolder, "descriptors/", "name of descriptorsFolder");
DEFINE_string(SIFTFolder, "SIFT/", "name of SIFT folder");
DEFINE_string(binaryFolder, "binaryFiles/", "Path to binary files");
DEFINE_string(
    dataPath, "/home/erik/Projects/3DscanData/DUC/Floor1",
    "Path to where the program should search for the various folders it needs");
DEFINE_int32(startIndex, 0, "Scan number to start with");
DEFINE_int32(startNumber, -1, "Tells the program what number to start with");
DEFINE_int32(numScans, -1,
             "Number of scans to place, default or -1 will cause all scans in "
             "the folder to placed");
DEFINE_int32(numLevels, 5, "Number of levels in the pyramid");
DEFINE_int32(metricNumber, 3,
             "Which metric version the algorithm uses for placement");
DEFINE_int32(top, -1, "Only shows the top x placements, -1=ALL");
DEFINE_int32(threads, 0,
             "Number of threads to use.  If 0 OMP runtime will decide");
DEFINE_double(
    scale, -1,
    "Scale used to size the density maps.  If -1, it will be looked up");

void prependDataPath() {
  FLAGS_floorPlan = FLAGS_dataPath + "/" + FLAGS_floorPlan;
  FLAGS_dmFolder = FLAGS_dataPath + "/" + FLAGS_dmFolder;
  FLAGS_outputV1 = FLAGS_dataPath + "/" + FLAGS_outputV1;
  FLAGS_outputV2 = FLAGS_dataPath + "/" + FLAGS_outputV2;
  FLAGS_zerosFolder = FLAGS_dataPath + "/" + FLAGS_zerosFolder;
  FLAGS_voxelFolder = FLAGS_dataPath + "/" + FLAGS_voxelFolder;
  FLAGS_rotFolder = FLAGS_dataPath + "/" + FLAGS_rotFolder;
  FLAGS_panoFolder = FLAGS_dataPath + "/" + FLAGS_panoFolder;
  FLAGS_PTXFolder = FLAGS_dataPath + "/" + FLAGS_PTXFolder;
  FLAGS_normalsFolder = FLAGS_dataPath + "/" + FLAGS_normalsFolder;
  FLAGS_descriptorsFolder = FLAGS_dataPath + "/" + FLAGS_descriptorsFolder;
  FLAGS_SIFTFolder = FLAGS_dataPath + "/" + FLAGS_SIFTFolder;
  FLAGS_binaryFolder = FLAGS_dataPath + "/" + FLAGS_binaryFolder;
}

void parseFolder(const std::string &name, std::vector<std::string> &out) {
  for (auto &file : folderToIterator(name))
    out.push_back(file.path().filename().string());
}

boost::filesystem::directory_iterator
folderToIterator(const std::string &name) {
  boost::filesystem::path folder(name);
  if (!boost::filesystem::exists(folder) ||
      !boost::filesystem::is_directory(folder)) {
    std::cout << folder << " does not exists" << std::endl;
    exit(1);
  }
  return boost::filesystem::directory_iterator(folder);
}

int numberToIndex(const std::vector<std::string> &names, const int number) {
  for (int i = 0; i < names.size(); ++i)
    if (number == std::stoi(names[i].substr(names[i].find(".") - 3, 3)))
      return i;
  std::cout << "Could not find scan with number: " << number << std::endl;
  exit(2);
}

/*int main() {
  prependDataPath();
  std::vector<std::string> test;
  parseFolder(FLAGS_zerosFolder, test);
  for (auto & n : test) {
    std::cout << n << std::endl;
  }
}*/