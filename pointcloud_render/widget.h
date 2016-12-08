#ifndef WIDGET_H
#define WIDGET_H

#include <QBasicTimer>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions_3_0>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>
#include <QtOpenGLExtensions>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <opencv/cv.hpp>

#include "framerecorder.h"
#include <scan_gflags.h>
// #include <scan_typedefs.hpp>
constexpr double PI = 3.14159265358979323846;

typedef pcl::PointXYZRGB PointType;

DECLARE_double(omega);
DECLARE_double(h_velocity);
DECLARE_double(h_min);
DECLARE_double(d_min);
DECLARE_double(d_velocity);
DECLARE_double(FPS);
DECLARE_double(ss_factor);

namespace Ui {
class Widget;
}

class Widget : public QOpenGLWidget, protected QOpenGLFunctions_3_0 {
  Q_OBJECT

public:
  explicit Widget(const std::string &name, const std::string &out_folder,
                  QWidget *parent = 0);
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

  double start_PI;

  enum states {
    pure_rotation,
    plane_down,
    zoom_in,
    zoom_out,
    plane_up,
    plane_final,
    zoom_out_and_plane_down,
    done
  };

  states current_state, state_after_spin;
  double dist_to_spin;

  void set_next_state();
  void do_state_outputs();

  cv::AutoBuffer<uchar> buffer;
  cv::Mat img;
  FrameRecorder recorder;
  std::vector<std::unique_ptr<QOpenGLBuffer>> vertex_buffers;
  std::unique_ptr<QOpenGLBuffer> aa_buffer, index_buffer;
  std::unique_ptr<QOpenGLFramebufferObject> render_fbo = nullptr,
                                            texture_fbo = nullptr;
  void allocate();

  std::unique_ptr<QOpenGLShaderProgram> cloud_program, aa_program;
  int vertex_location, color_location, position_location, sampler_location,
      texcoord_location, viewport_location, planes_location;
  QMatrix4x4 projection, mvp;
  std::vector<long> buffer_sizes;
  int aa_width, aa_height;
  int aa_factor = 2;

  class ClippingCube {
  private:
    typedef Eigen::AlignedBox<double, 3> AB;
    static constexpr int num_corners = 7, num_planes = 6;

  public:
    ClippingCube(Eigen::Vector3d &bl_corner, Eigen::Vector3d &tr_corner) {
      AB aligned_box(bl_corner, tr_corner);
      set_corners(aligned_box);
    };

    ClippingCube(Eigen::Vector3d &&bl_corner, Eigen::Vector3d &&tr_corner) {
      AB aligned_box(bl_corner, tr_corner);
      set_corners(aligned_box);
    };

    inline bool is_in(PointType &p) {
      if (!_active)
        return false;

      Eigen::Vector3d point(p.x, p.z, p.y);

      bool is_in = true;
      for (int i = 0; i < num_planes; ++i)
        if (normals[i].dot(point) + k[i] < 0)
          is_in = false;

      double min_dist = normals[0].dot(point) + k[0];
      for (int i = 1; i < num_planes; ++i)
        min_dist = std::min(min_dist, normals[i].dot(point) + k[i]);

      return min_dist > 0.0;
    }

    void activate(void) { _active = true; };

    void deactivate(void) { _active = false; };

    void rotate(const Eigen::Matrix3d &mat) {
      const Eigen::Vector3d _center = center();
      for (int i = 0; i < num_corners; ++i)
        corners[i] = mat * (corners[i] - _center) + _center;

      calc_planes();
    }

    void rotate(const double theta) {
      Eigen::Matrix3d rot_mat =
          Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
      rotate(rot_mat);
    }

    inline Eigen::Vector3d center(void) {
      return (corners[3] + corners[4]) / 2.0;
    }

    void grow(double amount) {
      for (int i = 0; i < num_corners; ++i) {
        Eigen::Vector3d direction = (corners[i] - center()).normalized();
        corners[i] += direction * amount;
      }
      calc_planes();
    }

    void grow(double amount, const Eigen::Vector3d &axis) {
      for (int i = 0; i < num_corners; ++i) {
        Eigen::Vector3d direction = (corners[i] - center());
        direction = (direction.array() * (axis.array())).matrix().normalized();
        corners[i] += direction * amount;
      }
      calc_planes();
    }

    void grow(double amount, const Eigen::Vector3d &&axis) {
      for (int i = 0; i < num_corners; ++i) {
        Eigen::Vector3d direction = (corners[i] - center());
        direction = (direction.array() * axis.array()).matrix().normalized();
        corners[i] += direction * amount / 2.0;
      }
      calc_planes();
    }

    void growX(double amount) { grow(amount, Eigen::Vector3d::UnitX()); }

    void growY(double amount) { grow(amount, Eigen::Vector3d::UnitY()); }

    void growZ(double amount) { grow(amount, Eigen::Vector3d::UnitZ()); }

    void growXZ(double amount) {
      grow(amount, Eigen::Vector3d::UnitX() + Eigen::Vector3d::UnitZ());
    }

    void translate(double amount, const Eigen::Vector3d &direction) {
      for (int i = 0; i < num_corners; ++i)
        corners[i] += direction * amount;

      calc_planes();
    }

    void set_center(const Eigen::Vector3d &new_center) {
      Eigen::Vector3d trans = new_center - center();
      translate(trans.norm(), trans.normalized());
    }

    void set_center(const Eigen::Vector3d &&new_center) {
      Eigen::Vector3d trans = new_center - center();
      translate(trans.norm(), trans.normalized());
    }

    std::vector<QVector4D> package_planes() {
      std::vector<QVector4D> res;
      if (_active) {
        for (int i = 0; i < num_planes; ++i) {
          res.emplace_back(normals[i][0], normals[i][1], normals[i][2], k[i]);
        }
      } else {
        for (int i = 0; i < num_planes; ++i) {
          res.emplace_back(0, 0, 0, -5);
        }
      }

      return res;
    }

  private:
    void set_corners(AB &aligned_box) {
      corners[0] = aligned_box.corner(AB::BottomLeftFloor);
      corners[1] = aligned_box.corner(AB::BottomRightFloor);
      corners[2] = aligned_box.corner(AB::TopLeftFloor);
      corners[3] = aligned_box.corner(AB::TopRightFloor);

      corners[4] = aligned_box.corner(AB::BottomLeftCeil);
      corners[5] = aligned_box.corner(AB::BottomRightCeil);
      corners[6] = aligned_box.corner(AB::TopLeftCeil);

      calc_planes();
    }

    void calc_planes() {
      normals[0] = calc_normal(corners[0], corners[1], corners[2]);
      k[0] = -normals[0].dot(corners[0]);

      normals[1] = calc_normal(corners[2], corners[3], corners[6]);
      k[1] = -normals[1].dot(corners[2]);

      normals[2] = calc_normal(corners[3], corners[1], corners[5]);
      k[2] = -normals[2].dot(corners[3]);

      normals[3] = calc_normal(corners[1], corners[0], corners[5]);
      k[3] = -normals[3].dot(corners[1]);

      normals[4] = calc_normal(corners[0], corners[2], corners[4]);
      k[4] = -normals[4].dot(corners[0]);

      normals[5] = calc_normal(corners[4], corners[5], corners[6]);
      k[5] = -normals[5].dot(corners[4]);
    }

    Eigen::Vector3d calc_normal(const Eigen::Vector3d &p1,
                                const Eigen::Vector3d &p2,
                                const Eigen::Vector3d &p3) {
      Eigen::Vector3d edge1 = p1 - p2;
      Eigen::Vector3d edge2 = p3 - p2;
      Eigen::Vector3d n = edge1.cross(edge2);

      if (std::acos(n.dot(center() - p1) /
                    ((center() - p1).norm() * n.norm())) > PI / 2.0)
        n *= -1.0;

      return n;
    }

    Eigen::Vector3d corners[num_corners];
    Eigen::Vector3d normals[num_planes];
    double k[num_planes];
    bool _active = true;
  };
  std::list<ClippingCube> cubes;

  bool is_in_cube(PointType &p);
  double height_max, height_min, h_scale;
  std::vector<long> h_bins;
  inline long binner(float h) {
    return std::max(0l,
                    std::min((long)h_bins.size(),
                             static_cast<long>((h - height_min) * h_scale)));
  }
};

#endif // WIDGET_H
