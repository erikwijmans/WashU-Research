#include "widget.h"
#include <QApplication>

DEFINE_string(cloud_name, "pointCloud.ply", "Name of pointcloud");

int main(int argc, char *argv[]) {

  QApplication a(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  const std::string out_folder =
      "./" + FLAGS_cloud_name.substr(0, FLAGS_cloud_name.find("."));

  Widget w(FLAGS_outputV2 + FLAGS_cloud_name, out_folder);
  w.setFixedSize(1920, 1080);
  w.show();

  return a.exec();
}
