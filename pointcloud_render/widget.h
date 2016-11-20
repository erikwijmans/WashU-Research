#ifndef WIDGET_H
#define WIDGET_H

#include <QBasicTimer>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_3_0>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QtOpenGLExtensions>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Eigen>
#include <opencv/cv.hpp>

#include "framerecorder.h"
#include <scan_gflags.h>

typedef pcl::PointXYZRGB PointType;

DECLARE_double(omega);
DECLARE_double(h_velocity);
DECLARE_double(h_min);
DECLARE_double(d_min);
DECLARE_double(d_velocity);
DECLARE_double(FPS);

namespace Ui {
class Widget;
}

class Widget : public QOpenGLWidget, protected QOpenGLFunctions_3_0 {
  Q_OBJECT

public:
  explicit Widget(const std::string &name, QWidget *parent = 0);
  ~Widget();

protected:
  void initializeGL();
  void paintGL();
  void resizeGL(int width, int height);

  void timerEvent(QTimerEvent *);
  void keyPressEvent(QKeyEvent *e);

private:
  Ui::Widget *ui;
  void draw();
  void set_matrices();
  void capture();

  pcl::PointCloud<PointType>::Ptr cloud;

  Eigen::Vector2d camera_origin, eye, rails_eye;
  Eigen::Vector2d org_camera, org_eye;
  double camera_y, eye_y;
  const Eigen::Vector3d k_up;

  Eigen::Array3d max, min;
  Eigen::Vector2d floor_box;

  void bounding_box();
  void filter();

  QBasicTimer timer;

  double h_v, d_v, e_v;

  long frame_counter;
  bool render;

  double distance, radians_traveled, start_distance;
  double h_clipping_plane;
  const double omega;
  long num_points_drawn;

  double start_PI;

  enum states { pure_rotation, plane_down, zoom_in, zoom_out, plane_up, done };

  states current_state, after_spin_state;
  double dist_to_spin;

  void set_next_state();
  void do_state_outputs();

  cv::AutoBuffer<uchar> buffer;
  cv::Mat img;
  FrameRecorder recorder;
  std::unique_ptr<QOpenGLBuffer> vertex_buffer;
  void allocate();

  std::unique_ptr<QOpenGLShaderProgram> program;
  int vertex_location, color_location;
  QMatrix4x4 projection;
  std::vector<int> h_bins;
  int binner(float y);
};

#endif // WIDGET_H
