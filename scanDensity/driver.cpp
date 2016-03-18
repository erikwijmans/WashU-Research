#include "scanDensity_scanDensity.h"


void saveImages(const std::vector<cv::Mat> & images, 
  const std::vector<std::string> & names);

int main(int argc, char *argv[]) {
  std::vector<std::string> names;
  cvNamedWindow("Preview", CV_WINDOW_NORMAL);

	DensityMaps maps (argc, argv);
	
  BoundingBox bBox(maps.getPoints());
  CloudAnalyzer2D analyzer2D (maps.getPoints(), &bBox, maps.getR());
  while(maps.hasNext()) {
    maps.run(false);
    bBox.run();

    analyzer2D.run(maps.getScale());
    analyzer2D.examinePointEvidence();
    analyzer2D.examineFreeSpaceEvidence();

    for(auto & image : analyzer2D.getPointEvidence()) {
      cv::imshow("Preview", image);
      cv::waitKey(0);
    }

    for(auto & image : analyzer2D.getFreeSpaceEvidence()) {
      cv::imshow("Preview", image);
      cv::waitKey(0);
    }

    names.clear();
    maps.get2DPointNames(names);
    names.clear();
    saveImages(analyzer2D.getPointEvidence(), names);
    maps.get2DFreeNames(names);
    saveImages(analyzer2D.getFreeSpaceEvidence(), names);

    maps.getNext();
  }
}

void saveImages(const std::vector<cv::Mat> & images, 
  const std::vector<std::string> & names) {
  for (int i = 0; i < names.size(); ++i)
    cv::imwrite(names[i], images[i]);
  
}