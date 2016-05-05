#include "scanDensity_scanDensity.h"
#include "scanDensity_3DInfo.h"

void saveImages(const std::vector<cv::Mat> & images,
  const std::vector<std::string> & names);
void saveZeroZero(const Eigen::Vector2i & zZ, const std::string & name);

static const double voxelsPerMeter = 20.0;

int main(int argc, char *argv[]) {

  cvNamedWindow("Preview", CV_WINDOW_NORMAL);

	DensityMapsManager manager (argc, argv);

  #pragma omp parallel shared(manager) if(!FLAGS_preview)
  {
    bool loop = true;
    while(loop) {
      bool twoD = false;
      bool threeD = false;
      std::vector<std::string> peNames, feNames;
      std::vector<std::string> pointNames, freeNames;
      std::string zerosName, metaDataName;
      DensityMapsManager::PointsPtr points;
      DensityMapsManager::MatPtr R;
      double scale;
      #pragma omp critical
      {
        if (!manager.hasNext()) {
          loop = false;
        } else {
          manager.run();
          points = manager.getPointsWithCenter();
          R = manager.getR();
          manager.get2DPointNames(peNames);
          manager.get2DFreeNames(feNames);
          manager.get3DPointNames(pointNames);
          manager.get3DFreeNames(freeNames);
          zerosName = manager.getZerosName();
          metaDataName = manager.getMetaDataName();
          scale = manager.getScale();
          manager.setNext();
          twoD = FLAGS_2D && (FLAGS_redo || !manager.exists2D());
          threeD = FLAGS_3D && (FLAGS_redo || !manager.exists3D());
        }
      }


      if (twoD) {
        auto bBox2D = BoundingBox::Create(points, Eigen::Vector3f (9.0, 9.0, 6.0));
        bBox2D->run();

        CloudAnalyzer2D analyzer2D (points, R, bBox2D);
        analyzer2D.initalize(scale);

        if (FLAGS_pe) {
          analyzer2D.examinePointEvidence();
          if (FLAGS_save) {
            saveImages(analyzer2D.getPointEvidence(), peNames);
            saveZeroZero(analyzer2D.getImageZeroZero(),
              zerosName);
          }
        }
        if (FLAGS_fe) {
          analyzer2D.examineFreeSpaceEvidence();

          if (FLAGS_save)
            saveImages(analyzer2D.getFreeSpaceEvidence(), feNames);
        }
      }

      if (threeD) {
        auto bBox3D = BoundingBox::Create(points, Eigen::Vector3f (10.0, 10.0, 6.0));
        bBox3D->run();
        Eigen::Vector3f pointMin, pointMax;
        bBox3D->getBoundingBox(pointMin, pointMax);

        voxel::CloudAnalyzer3D analyzer3D (points,
          R, bBox3D);
        analyzer3D.run(voxelsPerMeter, scale);

        if (FLAGS_save)
          analyzer3D.saveVoxelGrids(pointNames, freeNames,
            metaDataName);
      }
    }
  }
}

void saveImages(const std::vector<cv::Mat> & images,
  const std::vector<std::string> & names) {
  for (int i = 0; i < names.size(); ++i)
    cv::imwrite(names[i], images[i]);
}

void saveZeroZero(const Eigen::Vector2i & zZ, const std::string & name) {
  std::ofstream out (name, std::ios::out | std::ios::binary);
  for (int i = 0; i < NUM_ROTS; ++i)
    out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
  out.close();
}