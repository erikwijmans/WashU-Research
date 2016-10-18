/**
  The scanDensity system is responsible for analyzing the pointclouds
  and creating the files needed by the placeScan system.
  Two types of evidence are examined in both 2D and 3D: point evidence
  and free space evidence
*/

#include "scanDensity_3DInfo.h"
#include "scanDensity_scanDensity.h"

#include <boost/progress.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <atomic>
#include <thread>

void saveImages(const std::vector<cv::Mat> &images,
                const std::vector<std::string> &names);
void saveZeroZero(const Eigen::Vector2i &zZ, const std::string &name);
void saveDoors(const std::vector<std::vector<place::Door>> &doors,
               const std::string &name);

static constexpr double voxelsPerMeter = 20.0;

int main(int argc, char *argv[]) {
  DensityMapsManager manager(argc, argv);

  boost::progress_display *show_progress = nullptr;
  if (FLAGS_quietMode)
    show_progress = new boost::progress_display(FLAGS_numScans);
  if (FLAGS_threads)
    omp_set_num_threads(FLAGS_threads);

  auto mtx = std::make_shared<boost::shared_mutex>();
  auto cv = std::make_shared<boost::condition_variable_any>();

  auto processed = std::make_shared<std::atomic_int>(2);

  while (manager.hasNext()) {
    auto $2DPointNames = std::make_shared<std::vector<std::string>>(),
         $2DFreeNames = std::make_shared<std::vector<std::string>>(),
         $3DPointNames = std::make_shared<std::vector<std::string>>(),
         $3DFreeNames = std::make_shared<std::vector<std::string>>();

    manager.run();
    auto $3DPoints = manager.getPointsWithCenter();
    auto $2DPoints = manager.getPointsNoCenter();
    auto R = manager.getR();
    auto doors = manager.getDoors();
    manager.get2DPointNames(*$2DPointNames);
    manager.get2DFreeNames(*$2DFreeNames);
    manager.get3DPointNames(*$3DPointNames);
    manager.get3DFreeNames(*$3DFreeNames);
    auto zerosName = std::make_shared<std::string>(manager.getZerosName());
    auto metaDataName =
        std::make_shared<std::string>(manager.getMetaDataName());
    auto doorsName = std::make_shared<std::string>(manager.getDoorsName());
    auto scale = manager.getScale();
    bool threeD = FLAGS_3D && (FLAGS_redo || !manager.exists3D());
    bool runDoors = FLAGS_redo || !manager.existsDoors();
    bool twoD = FLAGS_2D && (FLAGS_redo || !manager.exists2D());
    manager.setNext();

    boost::unique_lock<boost::shared_mutex> lock(*mtx);
    cv->wait(lock, [&processed]() { return processed->load() == 2; });
    *processed = 0;

    std::thread T1([=]() {
      boost::shared_lock<boost::shared_mutex> lock(*mtx);

      if (twoD || runDoors) {
        auto bBox2D =
            BoundingBox::Create($2DPoints, Eigen::Vector3f(9.0, 9.0, 6.0));
        bBox2D->run();
        CloudAnalyzer2D analyzer2D($3DPoints, R, bBox2D, doors);
        analyzer2D.initalize(scale);

        if (runDoors) {
          analyzer2D.rotateDoors();
          saveDoors(analyzer2D.getRotatedDoors(), *doorsName);
        }

        if (FLAGS_pe) {
          analyzer2D.examinePointEvidence();
          if (FLAGS_save) {
            saveImages(analyzer2D.getPointEvidence(), *$2DPointNames);
            saveZeroZero(analyzer2D.getImageZeroZero(), *zerosName);
          }
        }
        if (FLAGS_fe) {
          analyzer2D.examineFreeSpaceEvidence();

          if (FLAGS_save)
            saveImages(analyzer2D.getFreeSpaceEvidence(), *$2DFreeNames);
        }
      }

      processed->fetch_add(1);
      lock.unlock();
      cv->notify_all();
    });
    T1.detach();

    std::thread T2([=]() {
      boost::shared_lock<boost::shared_mutex> lock(*mtx);

      if (threeD) {
        auto bBox3D =
            BoundingBox::Create($3DPoints, Eigen::Vector3f(10.0, 10.0, 6.0));
        bBox3D->run();
        Eigen::Vector3f pointMin, pointMax;
        bBox3D->getBoundingBox(pointMin, pointMax);

        voxel::CloudAnalyzer3D analyzer3D($3DPoints, R, bBox3D);
        analyzer3D.run(voxelsPerMeter, scale);

        if (FLAGS_save)
          analyzer3D.saveVoxelGrids(*$3DPointNames, *$3DFreeNames,
                                    *metaDataName);
      }
      processed->fetch_add(1);

      lock.unlock();
      cv->notify_all();
    });
    T2.detach();

    if (show_progress)
      ++(*show_progress);
  }

  boost::unique_lock<boost::shared_mutex> lock(*mtx);
  cv->wait(lock, [&processed]() { return processed->load() == 2; });

  if (show_progress)
    delete show_progress;

  return 0;
}

void saveImages(const std::vector<cv::Mat> &images,
                const std::vector<std::string> &names) {
  assert(names.size() == images.size());
  for (int i = 0; i < names.size(); ++i)
    cv::imwrite(names[i], images[i]);
}

void saveZeroZero(const Eigen::Vector2i &zZ, const std::string &name) {
  std::ofstream out(name, std::ios::out | std::ios::binary);
  for (int i = 0; i < NUM_ROTS; ++i)
    out.write(reinterpret_cast<const char *>(zZ.data()), sizeof(zZ));
  out.close();
}

void saveDoors(const std::vector<std::vector<place::Door>> &doors,
               const std::string &name) {
  std::ofstream out(name, std::ios::out | std::ios::binary);
  for (auto &ds : doors) {
    int num = ds.size();
    out.write(reinterpret_cast<const char *>(&num), sizeof(num));
    for (auto &d : ds)
      d.writeToFile(out);
  }

  out.close();
}