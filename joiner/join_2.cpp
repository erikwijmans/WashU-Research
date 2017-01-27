#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/keyboard_event.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

DEFINE_double(x, 0, "bleh");

DEFINE_double(y, 0, "bleh");

DEFINE_double(z, 0, "bleh");

DEFINE_double(theta, 0, "bleh");

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointNormal NormalType;

class MyPointRepresentation : public pcl::PointRepresentation<NormalType> {
  using pcl::PointRepresentation<NormalType>::nr_dimensions_;

public:
  MyPointRepresentation() {
    // Define the number of dimensions
    nr_dimensions_ = 7;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray(const NormalType &p, float *out) const {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.normal_x;
    out[4] = p.normal_y;
    out[5] = p.normal_z;
    out[6] = p.curvature;
  }
};

pcl::visualization::PCLVisualizer::Ptr
rgbVis(pcl::PointCloud<PointType>::ConstPtr cloud) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
  viewer->addPointCloud<PointType>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  viewer->registerKeyboardCallback(
      [&, viewer](const pcl::visualization::KeyboardEvent &kb) {
        pcl::visualization::Camera c;
        viewer->getCameraParameters(c);

        Eigen::Map<Eigen::Vector3d> focal(c.focal);
        Eigen::Map<Eigen::Vector3d> pos(c.pos);
        Eigen::Vector3d view = (focal - pos).normalized();
        view[2] = 0;
        view.normalize();
        const double incMag = (kb.isShiftPressed() ? 1.0 : 0.5);

        Eigen::Vector3d incDir = incMag * view;
        Eigen::Vector3d perpInc =
            incMag * Eigen::Vector3d(-view[1], view[0], view[2]);

        if (kb.getKeySym() == "Up") {
          if (!kb.isCtrlPressed())
            focal += incDir;
          pos += incDir;
        }
        if (kb.getKeySym() == "Down") {
          if (!kb.isCtrlPressed())
            focal -= incDir;
          pos -= incDir;
        }
        if (kb.getKeySym() == "Left") {
          if (!kb.isCtrlPressed())
            focal += perpInc;
          pos += perpInc;
        }
        if (kb.getKeySym() == "Right") {
          if (!kb.isCtrlPressed())
            focal -= perpInc;
          pos -= perpInc;
        }
        viewer->setCameraParameters(c);
      });
  viewer->setCameraPosition(1, 0, 0, -1, 0, 0, 0, 0, 1);
  return (viewer);
}

std::tuple<Eigen::Array3f, Eigen::Array3f>
createPCLPointCloud(std::list<scan::PointXYZRGBA> &points,
                    pcl::PointCloud<PointType>::Ptr &cloud,
                    const Eigen::Matrix3d &rotMat,
                    const Eigen::Vector3d &trans);
pcl::PointCloud<NormalType>::Ptr
subsample_normals(const pcl::PointCloud<PointType>::Ptr &cloud);

constexpr double targetNumPoints = 100e6;
constexpr double startScale = 0.013;

bool sanity_check(const Eigen::Matrix4f &T) {
  for (int i = 0; i < 2; ++i)
    if (std::abs(T(i, 3)) >= 0.2)
      return false;

  if (std::abs(T(2, 3)) >= 0.1)
    return false;

  Eigen::Vector3f x, y, z;

  for (int j = 0; j < 3; ++j) {
    x[j] = T(0, j);
    y[j] = T(1, j);
    z[j] = T(2, j);
  }

  if (std::acos(std::abs(x.dot(Eigen::Vector3f::UnitX()))) >= 0.05)
    return false;

  if (std::acos(std::abs(y.dot(Eigen::Vector3f::UnitY()))) >= 0.05)
    return false;

  if (std::acos(std::abs(z.dot(Eigen::Vector3f::UnitZ()))) >= 0.05)
    return false;

  return true;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  pcl::UniformSampling<PointType> uniform_sampling;
  pcl::PointCloud<PointType>::Ptr target(new pcl::PointCloud<PointType>),
      source(new pcl::PointCloud<PointType>),
      output(new pcl::PointCloud<PointType>);

#if 1
#if 1
  pcl::io::loadPLYFile("cse_double.ply", *target);
  pcl::io::loadPLYFile("/home/erik/Projects/currentBest/CSE/Floor5/"
                       "placementOptions/V2/pointCloud.ply",
                       *source);
#else
  pcl::io::loadPLYFile("cse_double_mini.ply", *target);
  for (auto &&p : *target) {
    p.r = 255;
    p.g = 0;
    p.b = 0;
  }

  pcl::io::loadPLYFile("cse5_mini.ply", *source);
  for (auto &&p : *source) {
    p.r = 0;
    p.g = 255;
    p.b = 0;
  }
#endif

#if 0

  uniform_sampling.setInputCloud(target);
  target = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
  uniform_sampling.setRadiusSearch(0.15);
  uniform_sampling.filter(*target);

  pcl::io::savePLYFileBinary("cse_double_mini.ply", *target);

  uniform_sampling.setInputCloud(source);
  source = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
  uniform_sampling.setRadiusSearch(0.15);
  uniform_sampling.filter(*source);

  pcl::io::savePLYFileBinary("cse5_mini.ply", *source);

  exit(0);
#endif

  {
    Eigen::Vector3d trans(FLAGS_x, FLAGS_y, FLAGS_z);
    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity();
    Ti(0, 3) += trans[0];
    Ti(1, 3) += trans[1];
    Ti(2, 3) += trans[2];

    Eigen::Transform<float, 3, Eigen::TransformTraits::Affine> Ti_2(
        Eigen::AngleAxis<float>(FLAGS_theta * PI / 180.,
                                Eigen::Vector3f::UnitZ()));
    Ti = Ti_2 * Ti;

    auto tmp = source;
    source = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*tmp, *source, Ti);
  }

#if 0
  bool run_icp = true;
  pcl::search::KdTree<NormalType>::Ptr tree_source(
      new pcl::search::KdTree<NormalType>),
      tree_target(new pcl::search::KdTree<NormalType>);

  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced
  // against x, y, and z
  float alpha[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues(alpha);

  pcl::GeneralizedIterativeClosestPoint<NormalType, NormalType> icp;
  icp.setPointRepresentation(
      boost::make_shared<const MyPointRepresentation>(point_representation));
  icp.setSearchMethodTarget(tree_target);
  icp.setSearchMethodSource(tree_source);

  auto source_with_normals = subsample_normals(source);
  auto target_with_normals = subsample_normals(target);

  icp.setInputSource(source_with_normals);
  icp.setInputTarget(target_with_normals);
  icp.setTransformationEpsilon(1e-6);
  icp.setMaxCorrespondenceDistance(1e-1);

  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
  auto icp_result = source_with_normals;
  icp.setMaximumIterations(2);
  std::list<Eigen::Matrix4f> prev_4;
  for (int i = 0; i < 30 && run_icp; ++i) {
    // save cloud for visualization purpose
    source_with_normals = icp_result;

    // Estimate
    icp.setInputSource(source_with_normals);
    icp.align(*icp_result);

    // accumulate transformation between each Iteration
    Ti = icp.getFinalTransformation() * Ti;
    std::cout << Ti << std::endl << std::endl;

    if (!sanity_check(Ti) || icp.getFitnessScore() > 20) {
      run_icp = false;
      break;
    }

    // if the difference between this transformation and the previous
    // one
    // is smaller than the threshold, refine the process by reducing
    // the maximal correspondence distance
    if (fabs((icp.getLastIncrementalTransformation() - prev).sum()) <
        icp.getTransformationEpsilon())
      icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() -
                                       0.001);

    prev = icp.getLastIncrementalTransformation();

    if (prev_4.size() == 4) {
      double ave = 0;
      for (auto &m : prev_4)
        ave += (Ti - m).norm();

      ave /= 4;

      if (ave < icp.getTransformationEpsilon())
        break;

      prev_4.pop_front();
    }
    prev_4.emplace_back(Ti);
  }
  std::cout << "ICP worked: " << run_icp << std::endl
            << "has converged: " << icp.hasConverged()
            << " score: " << icp.getFitnessScore() << std::endl
            << "transformation: " << std::endl
            << Ti << std::endl
            << std::endl;

  if (run_icp) {
    auto tmp = source;
    source = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*tmp, *source, Ti);
  }

  output->insert(output->end(), target->begin(), target->end());
  target = nullptr;
  output->insert(output->end(), source->begin(), source->end());
  source = nullptr;

  pcl::io::savePLYFileBinary("double.ply", *output);

#else
  output->insert(output->end(), target->begin(), target->end());
  output->insert(output->end(), source->begin(), source->end());
#endif

  pcl::io::savePLYFileBinary("cse_triple.ply", *output);
  return 0;
#endif
  pcl::PointCloud<PointType>::Ptr ss = output;

  // uniform_sampling.setInputCloud(output);

  // uniform_sampling.setRadiusSearch(0.01);
  // uniform_sampling.filter(*ss);

  pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(ss);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}

pcl::PointCloud<NormalType>::Ptr
subsample_normals(const pcl::PointCloud<PointType>::Ptr &cloud) {
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  pcl::PointCloud<PointType>::Ptr ss(new pcl::PointCloud<PointType>);
  uniform_sampling.setRadiusSearch(0.04);
  uniform_sampling.filter(*ss);

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  pcl::search::KdTree<PointType>::Ptr tree(
      new pcl::search::KdTree<PointType>());

  norm_est.setSearchSurface(cloud);
  norm_est.setSearchMethod(tree);
  norm_est.setKSearch(30);

  pcl::PointCloud<NormalType>::Ptr output(new pcl::PointCloud<NormalType>);
  norm_est.setInputCloud(ss);
  norm_est.compute(*output);
  pcl::copyPointCloud(*ss, *output);

  std::cout << output->size() << std::endl;
  return output;
}
