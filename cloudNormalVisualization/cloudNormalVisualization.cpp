#include <dirent.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;
using namespace pcl;

void createBoundingBox(float *, float *, const vector<Vector3f> &);

unsigned int text_id = 0;
void keyboardEventOccurred(const visualization::KeyboardEvent &event,
                           void *viewer_void) {
  boost::shared_ptr<visualization::PCLVisualizer> viewer =
      *static_cast<boost::shared_ptr<visualization::PCLVisualizer> *>(
          viewer_void);
  if (event.getKeySym() == "r" && event.keyDown()) {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i) {
      sprintf(str, "text#%03d", i);
      viewer->removeShape(str);
    }
    text_id = 0;
  }
}

void mouseEventOccurred(const visualization::MouseEvent &event,
                        void *viewer_void) {
  boost::shared_ptr<visualization::PCLVisualizer> viewer =
      *static_cast<boost::shared_ptr<visualization::PCLVisualizer> *>(
          viewer_void);
  if (event.getButton() == visualization::MouseEvent::LeftButton &&
      event.getType() == visualization::MouseEvent::MouseButtonRelease) {
    std::cout << "Left mouse button released at position (" << event.getX()
              << ", " << event.getY() << ")" << std::endl;

    char str[512];
    sprintf(str, "text#%03d", text_id++);
    viewer->addText("clicked here", event.getX(), event.getY(), str);
  }
}

boost::shared_ptr<visualization::PCLVisualizer> normalsVis(
    PointCloud<PointXYZRGB>::ConstPtr cloud,
    PointCloud<Normal>::ConstPtr normals) {
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  boost::shared_ptr<visualization::PCLVisualizer> viewer(
      new visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
      visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<PointXYZRGB, Normal>(cloud, normals, 10, 0.05,
                                                    "normals");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return (viewer);
}

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    cout << "usage: cloudNormalVizualization <RGBZXY>.ptx " << endl;
    exit(0);
  }

  const char *ptxFileName = argv[1];

  PointCloud<Normal>::Ptr cloud_normals(new PointCloud<Normal>);
  PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
  boost::shared_ptr<visualization::PCLVisualizer> viewer;
  cloud->is_dense = true;

  vector<PointXYZRGB> cloudTmp;
  vector<Vector3f> points;
  ifstream ptxFile(ptxFileName, ios::in);
  int columns, rows;
  ptxFile >> columns >> rows;
  cout << rows * columns << endl;
  string line;
  for (int i = 0; i < 9; ++i) {
    getline(ptxFile, line);
  }

  for (int i = 0; i < rows * columns; ++i) {
    Vector3f point;
    ptxFile >> point[0] >> point[1] >> point[2];
    double itmp;
    ptxFile >> itmp;
    int rgb[3];
    ptxFile >> rgb[0] >> rgb[1] >> rgb[2];

    if (itmp < 0.2) continue;
    if (i % 10 != 0) continue;
    if (point[0] == 0 || point[1] == 0 || point[2] == 0) continue;

    PointXYZRGB tmp;
    tmp.x = point[0];
    tmp.y = point[1];
    tmp.z = point[2];
    tmp.r = static_cast<uint8_t>(rgb[0]);
    tmp.g = static_cast<uint8_t>(rgb[1]);
    tmp.b = static_cast<uint8_t>(rgb[2]);

    cloudTmp.push_back(tmp);
    points.push_back(point);
  }

  float pointMin[3], pointMax[3];
  createBoundingBox(pointMin, pointMax, points);

  for (int i = 0; i < cloudTmp.size(); ++i) {
    if (cloudTmp[i].x < pointMin[0] || cloudTmp[i].x > pointMax[0]) continue;
    if (cloudTmp[i].y < pointMin[1] || cloudTmp[i].y > pointMax[1]) continue;
    if (cloudTmp[i].z < pointMin[2] || cloudTmp[i].z > pointMax[2]) continue;

    cloud->points.push_back(cloudTmp[i]);
  }

  ptxFile.close();

  cout << cloud->points.size() << endl;

  // Create the normal estimation class, and pass the input dataset to it
  NormalEstimationOMP<PointXYZRGB, Normal> ne;
  ne.setInputCloud(cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation
  // object.
  // Its content will be filled inside the object, based on the given input
  // dataset (as no other search surface is given).
  search::KdTree<PointXYZRGB>::Ptr tree(new search::KdTree<PointXYZRGB>());
  ne.setSearchMethod(tree);

  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch(0.03);

  // Compute the features
  ne.compute(*cloud_normals);

  viewer = normalsVis(cloud, cloud_normals);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  return 0;
}

void createBoundingBox(float *pointMin, float *pointMax,
                       const vector<Vector3f> &points) {
  double averageX, averageY, sigmaX, sigmaY, averageZ, sigmaZ;
  averageX = averageY = sigmaX = sigmaY = averageZ = sigmaZ = 0;

  for (auto &point : points) {
    averageX += point[0];
    averageY += point[1];
    averageZ += point[2];
  }
  averageX = averageX / points.size();
  averageY = averageY / points.size();
  averageZ = averageZ / points.size();

  for (auto &point : points) {
    sigmaX += (point[0] - averageX) * (point[0] - averageX);
    sigmaY += (point[1] - averageY) * (point[1] - averageY);
    sigmaZ += (point[2] - averageZ) * (point[2] - averageZ);
  }
  sigmaX = sigmaX / (points.size() - 1);
  sigmaY = sigmaY / (points.size() - 1);
  sigmaZ = sigmaZ / (points.size() - 1);
  sigmaX = sqrt(sigmaX);
  sigmaY = sqrt(sigmaY);
  sigmaZ = sqrt(sigmaZ);

  double dX = 1.1 * 7 * sigmaX;
  double dY = 1.1 * 7 * sigmaX;
  double dZ = 1.1 * 5 * sigmaZ;

  pointMin[0] = averageX - dX / 2;
  pointMin[1] = averageY - dY / 2;
  pointMin[2] = averageZ - dZ / 2;

  pointMax[0] = averageX + dX / 2;
  pointMax[1] = averageY + dY / 2;
  pointMax[2] = averageZ + dZ / 2;
}