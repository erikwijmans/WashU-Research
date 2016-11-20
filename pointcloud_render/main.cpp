#include "widget.h"
#include <QApplication>

int main(int argc, char *argv[]) {

  QApplication a(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prependDataPath();

  Widget w(FLAGS_outputV2 + "pointCloud.ply");
  w.setFixedSize(1920, 1080);
  w.show();

  return a.exec();
}
