#include "widget.h"
#include "ui_widget.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#include <GL/gl.h>
#include <GL/glu.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>

DEFINE_double(omega, 0.002, "Spin rate in radians per frame");
DEFINE_double(h_velocity, 0, "Velocity of the y clipping plane");
DEFINE_double(d_velocity, 0, "Velocity of the distance from the center");
DEFINE_double(h_min, 0, "Min height of the h_clipping_plane");
DEFINE_double(d_min, 10, "Min distance");
DEFINE_double(FPS, 1, "target fps");

constexpr double h_bin_size = 0.01, h_bin_scale = 1. / h_bin_size;

constexpr double xRot = 0, yRot = 0, zRot = 0;

constexpr double fov = 60;

constexpr double PI = 3.141592653589793;

static const Eigen::Vector3d look_vector_start =
    Eigen::Vector3d(0, 3, 8).normalized();

struct VertexData {
  float data[6];
};

int Widget::binner(float y) {
  return std::max(
      0, std::min(static_cast<int>(std::round(h_bin_scale * (y - min[1]))),
                  static_cast<int>(h_bins.size() - 1)));
}

Widget::Widget(const std::string &name, QWidget *parent)
    : QOpenGLWidget(parent), ui(new Ui::Widget),
      cloud{new pcl::PointCloud<PointType>}, k_up{0, 1, 0}, frame_counter{0},
      render{false}, radians_traveled{0}, omega{FLAGS_omega / FLAGS_FPS},
      current_state{pure_rotation}, start_PI{0}, h_v{0}, d_v{0}, e_v{0},
      recorder{"."}, dist_to_spin{PI / 2.0}, after_spin_state{plane_down} {
  ui->setupUi(this);

  pcl::io::loadPLYFile(name, *cloud);

  // filter();
  bounding_box();

  // timer.start(1000 / FPS, this);
}

Widget::~Widget() { delete ui; }

void Widget::allocate() {
  std::cout << "allocating" << std::endl;

  std::sort(cloud->begin(), cloud->end(),
            [](PointType &a, PointType &b) { return a.z < b.z; });

  h_bins.resize(h_bin_scale * (cloud->at(cloud->size() - 1).z - min[1]) + 1, 0);
  std::vector<VertexData> points;
  int bin_index = binner(cloud->at(0).z);
  for (auto &p : *cloud) {
    VertexData tmp;
    tmp.data[0] = p.x;
    tmp.data[1] = p.z;
    tmp.data[2] = p.y;

    tmp.data[3] = p.r / 255.;
    tmp.data[4] = p.g / 255.;
    tmp.data[5] = p.b / 255.;

    points.emplace_back(tmp);

    int new_bin_index = binner(p.z);
    if (new_bin_index != bin_index) {
      for (int i = bin_index; i < new_bin_index; ++i)
        h_bins[i] = points.size() - 1;

      bin_index = new_bin_index;
    }
  }
  for (int i = bin_index; i < h_bins.size(); ++i)
    h_bins[i] = points.size();

  std::cout << points.size() << std::endl;

  vertex_buffer = std::unique_ptr<QOpenGLBuffer>(
      new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer));
  vertex_buffer->create();
  vertex_buffer->bind();
  vertex_buffer->allocate(points.data(), sizeof(VertexData) * points.size());
  vertex_buffer->release();
}

void Widget::initializeGL() {
  initializeOpenGLFunctions();
  glClearColor(0, 0, 0, 0);

  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_MULTISAMPLE);
  // glEnable(GL_POINT_SMOOTH);
  // glEnable(GL_LIGHTING);

  glEnable(GL_PROGRAM_POINT_SIZE);
  constexpr double point_size_init = 3.0;
  glPointSize(point_size_init);

  GLfloat attenuations_params[] = {1, 2, 1};

  // glPointParameterf(GL_POINT_SIZE_MIN, 1.0);
  glPointParameterf(GL_POINT_SIZE_MAX, 10.0);
  glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 0.5);
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, attenuations_params);

  program = std::unique_ptr<QOpenGLShaderProgram>(
      new QOpenGLShaderProgram(context()));

  program->addShaderFromSourceCode(
      QOpenGLShader::Vertex, "uniform mat4 mvp_matrix;\n"
                             "attribute vec3 vertex;\n"
                             "attribute vec3 color;\n"
                             "varying vec3 v_color;\n"
                             "void main() {\n"
                             "  v_color = color;\n"
                             "  gl_Position = mvp_matrix * vec4(vertex, 1.0);\n"
                             "}");
  program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                   "varying vec3 v_color;\n"
                                   "void main() {\n"
                                   "  gl_FragColor = vec4(v_color, 1.0);\n"
                                   "}");
  program->link();
  program->bind();

  vertex_location = program->attributeLocation("vertex");
  color_location = program->attributeLocation("color");

  allocate();
}

void Widget::set_next_state() {
  switch (current_state) {
  case pure_rotation:
    if (radians_traveled - start_PI >= dist_to_spin)
      current_state = after_spin_state;
    break;

  case plane_down:
    if (h_clipping_plane <= FLAGS_h_min)
      current_state = zoom_in;
    break;

  case zoom_in:
    if (distance <= FLAGS_d_min) {
      current_state = pure_rotation;
      start_PI = radians_traveled;
      after_spin_state = zoom_out;
    }
    break;

  case zoom_out:
    if (distance >= start_distance)
      current_state = plane_up;
    break;

  case plane_up:
    if (h_clipping_plane >= max[1]) {
      current_state = pure_rotation;
      start_PI = radians_traveled;
      dist_to_spin = PI / 6.0;
      after_spin_state = done;
    }
    break;

  default:
    break;
  }
}

void Widget::do_state_outputs() {
  switch (current_state) {
  case pure_rotation:
    e_v = (e_v + omega) / 2.0;
    break;

  case plane_down:
    h_v = std::max(
        0.5 * std::min(FLAGS_h_velocity, (h_clipping_plane - FLAGS_h_min)) +
            0.5 * h_v,
        0.1 * FLAGS_h_velocity);
    h_clipping_plane -= h_v / FLAGS_FPS;
    break;

  case zoom_in:
    d_v = std::max(0.5 * std::min(FLAGS_d_velocity, (distance - FLAGS_d_min)) +
                       0.5 * d_v,
                   0.1 * FLAGS_d_velocity);
    distance -= d_v / FLAGS_FPS;
    eye_y -= d_v / FLAGS_FPS * std::abs(look_vector_start[1]);
    break;

  case zoom_out:
    d_v =
        std::max(0.5 * std::min(FLAGS_d_velocity, (start_distance - distance)) +
                     0.5 * d_v,
                 0.1 * FLAGS_d_velocity);
    distance += 2 * d_v / FLAGS_FPS;
    eye_y += 2 * d_v / FLAGS_FPS * std::abs(look_vector_start[1]);
    break;

  case plane_up:
    h_v =
        std::max(0.5 * std::min(FLAGS_h_velocity, (max[1] - h_clipping_plane)) +
                     0.5 * h_v,
                 0.1 * FLAGS_h_velocity);
    h_clipping_plane += 2 * FLAGS_h_velocity / FLAGS_FPS;
    break;

  case done:
    if (render == true)
      recorder.exit();
    render = false;
    std::cout << "DONE" << std::endl;
    break;

  default:
    break;
  }
}

void Widget::set_matrices() {

  Eigen::Vector2d look_vector = rails_eye - camera_origin;
  look_vector.normalize();

  eye = camera_origin + distance * look_vector;

  if (render) {
    Eigen::Vector2d perp_vector(-look_vector[1], look_vector[0]);
    perp_vector.normalize();

    const double eye_distance = (eye - camera_origin).norm();
    eye += e_v * eye_distance * perp_vector;

    const double rails_distance = (rails_eye - camera_origin).norm();
    rails_eye += e_v * rails_distance * perp_vector;

    radians_traveled += e_v;
  }
  QMatrix4x4 matrix;
  matrix.lookAt(QVector3D(eye[0], eye_y, eye[1]),
                QVector3D(camera_origin[0], camera_y, camera_origin[1]),
                QVector3D(0, 1, 0));

  program->setUniformValue("mvp_matrix", projection * matrix);
}

void Widget::capture() {
  if (!render || !FLAGS_save)
    return;

  uchar *buf = buffer;
  glReadPixels(0, 0, width(), height(), GL_RGB, GL_UNSIGNED_BYTE, buf);

  for (int j = 0; j < height(); ++j) {
    uchar *dst = img.ptr<uchar>(height() - 1 - j);
    for (int i = 0; i < width(); ++i) {
      dst[3 * i + 0] = buf[3 * (j * width() + i) + 2];
      dst[3 * i + 1] = buf[3 * (j * width() + i) + 1];
      dst[3 * i + 2] = buf[3 * (j * width() + i) + 0];
    }
  }

  recorder.submit_frame(img.clone());
}

void Widget::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  set_matrices();
  draw();
  capture();

  set_next_state();
  do_state_outputs();

  if (render)
    update();
}

void Widget::resizeGL(int width, int height) {
  glViewport(0, 0, width, height);
  buffer.allocate(width * height * 3);
  img = cv::Mat(height, width, CV_8UC3);

  projection.setToIdentity();
  // Set perspective projection
  projection.perspective(fov, width / static_cast<double>(height), 1 / 10.,
                         200.);
}

void Widget::draw() {
  vertex_buffer->bind();

  quintptr offset = 0;

  program->enableAttributeArray(vertex_location);
  program->setAttributeBuffer(vertex_location, GL_FLOAT, offset, 3,
                              sizeof(VertexData));
  offset += 3 * sizeof(float);

  program->enableAttributeArray(color_location);
  program->setAttributeBuffer(color_location, GL_FLOAT, offset, 3,
                              sizeof(VertexData));

  glDrawArrays(GL_POINTS, 0, h_bins[binner(h_clipping_plane)]);
}

void Widget::bounding_box() {
  Eigen::Array3d average = Eigen::Array3d::Zero();
  for (auto &point : *cloud)
    average += point.getVector3fMap().cast<double>().array();

  average /= cloud->size();

  Eigen::Array3d sigma = Eigen::Array3d::Zero();

  for (auto &point : *cloud) {
    auto tmp = point.getVector3fMap().cast<double>().array();
    sigma += (tmp - average) * (tmp - average);
  }

  sigma /= cloud->size() - 1;
  sigma.sqrt();

  static const Eigen::Array3d delta(1, 1, 2.5);

  max = average + delta * sigma / 2.0;
  min = average - delta * sigma / 2.0;

  double tmp = max[1];
  max[1] = max[2];
  max[2] = tmp;

  tmp = min[1];
  min[1] = min[2];
  min[2] = tmp;

  floor_box =
      Eigen::Vector2d(std::abs(max[0] - min[0]), std::abs(max[2] - min[2]));

  distance = sqrt(sigma.square().sum()) / 6.0;
  start_distance = distance;

  camera_origin = Eigen::Vector2d(average[0], average[1]);
  camera_y = 0;

  Eigen::Vector3d start_eye =
      Eigen::Vector3d(average[0], 0, average[1]) + distance * look_vector_start;

  eye = Eigen::Vector2d(start_eye[0], start_eye[2]);
  eye_y = start_eye[1];
  rails_eye = eye;

  org_camera = camera_origin;
  org_eye = eye;

  h_clipping_plane = max[1];
}

void Widget::timerEvent(QTimerEvent *) { update(); }

void Widget::filter() {
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
  uniform_sampling.setRadiusSearch(1);
  uniform_sampling.filter(*cloud);
  std::cout << cloud->size() << std::endl;
}

void Widget::keyPressEvent(QKeyEvent *e) {
  if (e->key() == Qt::Key_S) {
    render = !render;
    if (render)
      update();
  } else if (e->key() == Qt::Key_R) {
    render = false;
    frame_counter = 0;

    camera_origin = org_camera;
    eye = org_eye;
    rails_eye = org_eye;

    radians_traveled = 0;
    h_clipping_plane = max[1];

    distance = start_distance;

    update();
  } else if (e->key() == Qt::Key_J) {
    distance += 1.0;
    update();
  } else if (e->key() == Qt::Key_K) {
    distance -= 1.0;
    update();
  }
}