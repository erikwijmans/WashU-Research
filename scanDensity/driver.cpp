#include "scanDensity_scanDensity.h"
#include "scanDensity_3DInfo.h"

void saveImages(const std::vector<cv::Mat> & images, 
  const std::vector<std::string> & names);
void saveZeroZero(const Eigen::Vector2i & zZ, const std::string & name);

static const double voxelsPerMeter = 20.0;

int main(int argc, char *argv[]) {
  
  cvNamedWindow("Preview", CV_WINDOW_NORMAL);

	DensityMapsManager manager (argc, argv);
	
  
  while(manager.hasNext()) {
    bool loaded = false;

    if (FLAGS_2D && (FLAGS_redo || !manager.exists2D())) {
      if(!loaded) {
        manager.run();
        loaded = true;
      }

      BoundingBox bBox2D(manager.getPointsNoCenter(), Eigen::Vector3f (9.0, 9.0, 6.0));
      bBox2D.run();

      CloudAnalyzer2D analyzer2D (manager.getPointsNoCenter(), manager.getR(), &bBox2D);
      analyzer2D.initalize(manager.getScale());

      if (FLAGS_pe) {
        analyzer2D.examinePointEvidence();
        std::vector<std::string> names;
        manager.get2DPointNames(names);
        saveImages(analyzer2D.getPointEvidence(), names);
        saveZeroZero(analyzer2D.getImageZeroZero(), 
          manager.getZerosName());
      }
      if (FLAGS_fe) {
        analyzer2D.examineFreeSpaceEvidence();
        std::vector<std::string> names;
        manager.get2DFreeNames(names);
        saveImages(analyzer2D.getFreeSpaceEvidence(), names);
      }
    }
    
    if (FLAGS_3D && (FLAGS_redo || !manager.exists3D())) {
      if(!loaded) manager.run();

      BoundingBox bBox3D (manager.getPointsWithCenter(), Eigen::Vector3f (7.0, 7.0, 6.0));
      bBox3D.run();

      voxel::CloudAnalyzer3D analyzer3D (manager.getPointsWithCenter(),
        manager.getR(), &bBox3D);
      analyzer3D.run(voxelsPerMeter, manager.getScale());

      std::vector<std::string> pointNames, freeNames;
      manager.get3DPointNames(pointNames);
      manager.get3DFreeNames(freeNames);

      analyzer3D.saveVoxelGrids(pointNames, freeNames,
        manager.getMetaDataName());
    }

    manager.setNext();
  }
}

void saveImages(const std::vector<cv::Mat> & images, 
  const std::vector<std::string> & names) {
  for (int i = 0; i < names.size(); ++i)
    cv::imwrite(names[i], images[i]);
  
}

void saveZeroZero(const Eigen::Vector2i & zZ, const std::string & name) {
  std::ofstream out (name, std::ios::out | std::ios::binary);
  for (int i = 0; i < NUM_ROTS; ++i) {
    out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
  }
}