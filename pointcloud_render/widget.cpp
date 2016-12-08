#include "widget.h"
#include "ui_widget.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <GL/gl.h>
#include <GL/glu.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/io/ply_io.h>

DEFINE_double(omega, 0.002, "Spin rate in radians per second");
DEFINE_double(h_velocity, 0, "Velocity of the y clipping plane in m/s");
DEFINE_double(d_velocity, 0, "Velocity of the distance from the center in m/s");
DEFINE_double(h_min, 0, "Min height of the h_clipping_plane");
DEFINE_double(d_min, 10, "Min distance");
DEFINE_double(FPS, 1, "target fps");
DEFINE_double(ss_factor, 0.5, "sub-sampling factor");

template <typename ArrayTypeA, typename ArrayTypeB>
double gaussianWeight(const ArrayTypeA &pos, const ArrayTypeB &s) {
  return std::exp(-(pos.square() / (2 * s.square())).sum());
}

constexpr double h_bin_size = 0.01, h_bin_scale = 1. / h_bin_size;

constexpr double xRot = 0, yRot = 0, zRot = 0;

constexpr double fov = 60;

static const Eigen::Vector3d look_vector_start =
    Eigen::Vector3d(0, 5, 8).normalized();

struct VertexData {
  float data[6];
};

Widget::Widget(const std::string &name, const std::string &out_folder,
               QWidget *parent)
    : QOpenGLWidget(parent), ui(new Ui::Widget),
      cloud{new pcl::PointCloud<PointType>}, k_up{0, 1, 0}, frame_counter{0},
      render{false}, radians_traveled{0}, omega{FLAGS_omega / FLAGS_FPS},
      current_state{pure_rotation}, start_PI{0}, h_v{0}, d_v{0}, e_v{0},
      recorder{out_folder, FLAGS_save}, dist_to_spin{PI / 2.},
      state_after_spin{zoom_in} {
  ui->setupUi(this);

  pcl::io::loadPLYFile(name, *cloud);
  std::cout << cloud->size() << std::endl;

  filter();
  bounding_box();

  // timer.start(1000 / FPS, this);

  render = true;
}

Widget::~Widget() { delete ui; }

void Widget::allocate() {
  std::sort(cloud->begin(), cloud->end(),
            [](PointType &p1, PointType &p2) { return p1.z < p2.z; });

  std::cout << "allocating" << std::endl;

  constexpr long max_buffer_size =
      static_cast<long>(std::numeric_limits<int>::max()) / 2;
  std::cout << "max buffer size: " << max_buffer_size << std::endl;

  h_clipping_plane = cloud->at(cloud->size() - 1).z - 5.0;
  std::vector<VertexData> points;
  long bytes_allocated = 0;
  long points_buffered = 0;
  long max_p = 0;

  std::function<void()> buffer_points = [&]() {
    const long bytes = points.size() * sizeof(VertexData);

    const long p = points.size();

    vertex_buffers.push_back(std::unique_ptr<QOpenGLBuffer>(
        new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer)));
    vertex_buffers.back()->create();
    vertex_buffers.back()->bind();
    vertex_buffers.back()->setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffers.back()->allocate(points.data(), bytes);
    vertex_buffers.back()->release();

    bytes_allocated += bytes;
    points_buffered += p;

    std::cout << "Buffered " << bytes << " bytes" << std::endl;

    max_p = std::max(max_p, p);

    buffer_sizes.push_back(p);
    points.clear();
  };

  h_scale = 5.0 / (FLAGS_h_velocity / FLAGS_FPS);
  height_max = cloud->at(cloud->size() - 1).z + 0.1;
  height_min = cloud->at(0).z - 0.1;
  h_bins.resize((height_max - height_min) * h_scale, 0);
  std::cout << h_bins.size() << std::endl;

  long count = 0;
  for (auto &p : *cloud) {
    VertexData tmp;
    tmp.data[0] = p.x;
    tmp.data[1] = p.z;
    tmp.data[2] = p.y;

    tmp.data[3] = p.r / 255.;
    tmp.data[4] = p.g / 255.;
    tmp.data[5] = p.b / 255.;

    points.emplace_back(tmp);

    if (points.size() * sizeof(VertexData) >=
        max_buffer_size - sizeof(VertexData)) {
      buffer_points();
    }
    h_bins[binner(p.z)] = ++count;
  }
  for (int i = binner(cloud->at(cloud->size() - 1).z); i < h_bins.size(); ++i) {
    h_bins[i] = count;
  }

  buffer_points();

  std::cout << cloud->size() << std::endl;
  std::cout << points_buffered << std::endl;
  std::cout << h_bins[h_bins.size() - 1] << std::endl;

  /*index_buffer = std::unique_ptr<QOpenGLBuffer>(
      new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer));
  index_buffer->create();
  index_buffer->bind();
  index_buffer->setUsagePattern(QOpenGLBuffer::DynamicDraw);
  index_buffer->allocate(max_p * sizeof(int));
  index_buffer->release();*/

  /* clang-format off */
  //                          x     y
  static float quadVertices[] = {
      // Positions   // TexCoords
      -1.0f,  1.0f,  /*0.0f, 1.0f,*/
      -1.0f, -1.0f,  /*0.0f, 0.0f,*/
       1.0f, -1.0f,  /*1.0f, 0.0f,*/

      -1.0f,  1.0f,  /*0.0f, 1.0f,*/
       1.0f, -1.0f,  /*1.0f, 0.0f,*/
       1.0f,  1.0f,  /*1.0f, 1.0f*/
  };
  /* clang-format on */

  aa_buffer = std::unique_ptr<QOpenGLBuffer>(
      new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer));
  aa_buffer->create();
  aa_buffer->bind();
  aa_buffer->setUsagePattern(QOpenGLBuffer::StaticDraw);
  aa_buffer->allocate(quadVertices, sizeof(quadVertices));
  aa_buffer->release();
}

void Widget::initializeGL() {
  initializeOpenGLFunctions();
  this->makeCurrent();
  glClearColor(0, 0, 0, 0);

  glEnable(GL_BLEND);
  // glBlendFunc(GL_ONE, GL_ONE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_MULTISAMPLE);
  // glEnable(GL_POINT_SMOOTH);
  // glEnable(GL_LIGHTING);
  glEnable(GL_ARB_texture_non_power_of_two);
  glEnable(GL_ARB_arrays_of_arrays);
  // glEnable(GL_CLIP_DISTANCE0);
  // glEnable(GL_CLIP_PLANE0);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glEnable(GL_TEXTURE_2D);

  glEnable(GL_PROGRAM_POINT_SIZE);
  constexpr double point_size_init = 1.5;
  glPointSize(point_size_init);

  GLfloat attenuations_params[] = {1, 2, 2};

  glPointParameterf(GL_POINT_SIZE_MIN, 0.2);
  // glPointParameterf(GL_POINT_SIZE_MAX, 10.0);
  glPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 0.25);
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, attenuations_params);

  aa_program = std::unique_ptr<QOpenGLShaderProgram>(
      new QOpenGLShaderProgram(context()));

  aa_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                      R"(
      attribute vec2 position, tex_coord;
      varying vec2 v_tex_coord;

      void main() {
        v_tex_coord = position * 0.5 + 0.5;
        gl_Position = vec4(position.x, position.y, 0.0, 1.0);
      })");
  aa_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                      R"(
                               uniform sampler2D texture;
                               varying vec2 v_tex_coord;
                               uniform vec2 viewport;
                               uniform float weights [25];
                               void main() {
                                 vec3 color = vec3(0.0);
                                 for (int j = 0; j < 5; j++) {
                                   for (int i = 0; i < 5; i++) {
                                     vec2 off = vec2(i - 2, j - 2)/viewport;
                                     color += texture2D(texture,
                               v_tex_coord + off).rgb * weights[j*5 + i];
                                   }
                                 }
                                 gl_FragColor = vec4(color, 1.0);
                               })");

  aa_program->link();
  aa_program->bind();
  position_location = aa_program->attributeLocation("position");
  sampler_location = aa_program->uniformLocation("texture");
  texcoord_location = aa_program->attributeLocation("tex_coord");
  viewport_location = aa_program->uniformLocation("viewport");
  float values[25];

  static const Eigen::Array2d sigma(0.5, 0.5);
  double sum = 0;
  for (int j = 0; j < 5; ++j) {
    for (int i = 0; i < 5; ++i) {
      double v = gaussianWeight(Eigen::Array2d(i - 2, j - 2), sigma);
      values[j * 5 + i] = v;
      sum += v;
    }
  }
  for (int i = 0; i < 25; ++i)
    values[i] /= sum;

  aa_program->setUniformValueArray("weights", values, 25, 1);
  aa_program->release();

  clipping_program = std::shared_ptr<QOpenGLShaderProgram>(
      new QOpenGLShaderProgram(context()));

  clipping_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                            R"(
      #define num_planes 6
      uniform mat4 mvp_matrix;
      uniform vec4 planes[num_planes];
      attribute vec3 vertex;
      attribute vec3 color;
      varying vec3 v_color;
      varying float clip_dist;
      void main() {
        vec4 vert4 = vec4(vertex, 1.0);
        float min_dst = dot(planes[0], vert4);
        for (int i = 1; i < num_planes; i++)
          min_dst = min(dot(planes[i], vert4), min_dst);

        v_color = color;
        clip_dist = -1.0 * min_dst;
        gl_Position = mvp_matrix * vert4;
      })");
  clipping_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                            R"(
                               varying vec3 v_color;
                               varying float clip_dist;
                               void main() {
                                if (clip_dist < 0.0)
                                  discard;
                                gl_FragColor = vec4(v_color, 1.0);
                               })");
  clipping_program->link();
  clipping_program->bind();

  planes_location = clipping_program->uniformLocation("planes");
  clipping_program->release();

  nonclipping_program = std::shared_ptr<QOpenGLShaderProgram>(
      new QOpenGLShaderProgram(context()));

  nonclipping_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                               R"(
      uniform mat4 mvp_matrix;
      attribute vec3 vertex;
      attribute vec3 color;
      varying vec3 v_color;

      void main() {
        vec4 vert4 = vec4(vertex, 1.0);

        v_color = color;
        gl_Position = mvp_matrix * vert4;
      })");
  nonclipping_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                               R"(
                               varying vec3 v_color;
                               void main() {
                                gl_FragColor = vec4(v_color, 1.0);
                               })");
  nonclipping_program->link();
  nonclipping_program->bind();
  nonclipping_program->release();

  allocate();
}

void Widget::set_next_state() {
  switch (current_state) {
  case pure_rotation:
    if (radians_traveled - start_PI >= dist_to_spin) {
      current_state = state_after_spin;

      cube.activate();
    }
    break;

  case plane_down:
    if (h_clipping_plane <= FLAGS_h_min && distance <= FLAGS_d_min)
      current_state = zoom_in;
    break;

  case zoom_in:
    if (distance <= FLAGS_d_min) {
      current_state = pure_rotation;
      start_PI = radians_traveled;
      dist_to_spin = PI;
      state_after_spin = zoom_out_and_plane_down;
    }
    break;

  case zoom_out:
    if (distance >= start_distance) {
      current_state = pure_rotation;
      start_PI = radians_traveled;
      dist_to_spin = PI / 3.0;
      state_after_spin = plane_final;

      cube.deactivate();
    }
    break;

  case plane_up:
    if (h_clipping_plane >= max[1])
      current_state = pure_rotation;

    break;

  case zoom_out_and_plane_down:
    if (h_clipping_plane <= FLAGS_h_min && distance >= start_distance) {
      current_state = pure_rotation;
      start_PI = radians_traveled;
      dist_to_spin = PI / 2.0;
      state_after_spin = plane_final;

      cube.deactivate();
    }

  case plane_final:
    if (h_clipping_plane < cloud->at(0).z)
      current_state = done;
    break;

  default:
    break;
  }
}

void Widget::do_state_outputs() {
  switch (current_state) {
  case pure_rotation:
    e_v = (e_v + omega) / 2.0;

    cube.rotate(-e_v);

    break;

  case plane_down:
    if (h_clipping_plane >= FLAGS_h_min) {
      h_v = std::max(
          0.5 * std::min(FLAGS_h_velocity, (h_clipping_plane - FLAGS_h_min)) +
              0.5 * h_v,
          0.1 * FLAGS_h_velocity);
      h_clipping_plane -= h_v / FLAGS_FPS;
    }

    if (h_clipping_plane <= cloud->at(cloud->size() - 1).z &&
        distance >= FLAGS_d_min) {
      d_v =
          std::max(0.5 * std::min(FLAGS_d_velocity, (distance - FLAGS_d_min)) +
                       0.5 * d_v,
                   0.1 * FLAGS_d_velocity);
      distance -= d_v / FLAGS_FPS;
      eye_y -= d_v / FLAGS_FPS * std::abs(look_vector_start[1]);
      camera_y += d_v / FLAGS_FPS * std::abs(look_vector_start[1]) / 2.0;
      camera_y = std::min(2.5, camera_y);

      cube.growXZ(-d_v / FLAGS_FPS / 1.5);
    }

    break;

  case zoom_out_and_plane_down:
    if (h_clipping_plane >= FLAGS_h_min) {
      h_v = std::max(
          0.5 * std::min(FLAGS_h_velocity, (h_clipping_plane - FLAGS_h_min)) +
              0.5 * h_v,
          0.1 * FLAGS_h_velocity);
      h_clipping_plane -= h_v / FLAGS_FPS;
    }

    if (h_clipping_plane <= cloud->at(cloud->size() - 1).z &&
        distance <= start_distance) {
      d_v = std::max(
          0.5 * std::min(FLAGS_d_velocity, (start_distance - distance)) +
              0.5 * d_v,
          0.1 * FLAGS_d_velocity);
      distance += d_v / FLAGS_FPS;
      eye_y += d_v / FLAGS_FPS * std::abs(look_vector_start[1]);

      cube.growXZ(-d_v / FLAGS_FPS / 1.5);
    }

    break;

  case zoom_in:
    d_v = std::max(0.5 * std::min(FLAGS_d_velocity, (distance - FLAGS_d_min)) +
                       0.5 * d_v,
                   0.1 * FLAGS_d_velocity);
    distance -= d_v / FLAGS_FPS;
    eye_y -= d_v / FLAGS_FPS * std::abs(look_vector_start[1]);
    camera_y += d_v / FLAGS_FPS * std::abs(look_vector_start[1]) / 2.0;
    camera_y = std::min(2.5, camera_y);

    cube.growXZ(-d_v / FLAGS_FPS / 1.5);
    break;

  case zoom_out:
    d_v =
        std::max(0.5 * std::min(FLAGS_d_velocity, (start_distance - distance)) +
                     0.5 * d_v,
                 0.1 * FLAGS_d_velocity);
    distance += 2 * d_v / FLAGS_FPS;
    eye_y += 2 * d_v / FLAGS_FPS * std::abs(look_vector_start[1]);

    cube.growXZ(-d_v / FLAGS_FPS / 1.5);
    break;

  case plane_up:
    h_v =
        std::max(0.5 * std::min(FLAGS_h_velocity, (max[1] - h_clipping_plane)) +
                     0.5 * h_v,
                 0.1 * FLAGS_h_velocity);
    h_clipping_plane += 2 * h_v / FLAGS_FPS;
    break;

  case done:
    if (render == true)
      recorder.exit();
    render = false;
    std::cout << "DONE" << std::endl;
    QApplication::quit();
    break;

  case plane_final:
    h_v = std::max(
        0.5 * std::min(FLAGS_h_velocity, (h_clipping_plane - FLAGS_h_min)) +
            0.5 * h_v,
        0.1 * FLAGS_h_velocity);
    h_clipping_plane -= 5.0 * h_v / FLAGS_FPS;
    e_v *= 0.995;
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

    cube.set_center(Eigen::Vector3d(eye[0], 0, eye[1]));

    const double rails_distance = (rails_eye - camera_origin).norm();
    rails_eye += e_v * rails_distance * perp_vector;

    radians_traveled += e_v;
  }
  QMatrix4x4 matrix;
  matrix.lookAt(QVector3D(eye[0], eye_y, eye[1]),
                QVector3D(camera_origin[0], camera_y, camera_origin[1]),
                QVector3D(0, 1, 0));

  mvp = projection * matrix;
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
  set_matrices();
  draw();
  capture();

  set_next_state();
  do_state_outputs();

  if (render)
    update();
}

void Widget::resizeGL(int width, int height) {

  buffer.allocate(width * height * 3);
  img = cv::Mat(height, width, CV_8UC3);

  aa_width = width * aa_factor;
  aa_height = height * aa_factor;

  projection.setToIdentity();
  // Set perspective projection
  projection.perspective(fov, width / static_cast<double>(height), 1 / 10.,
                         200.);
}

void Widget::set_cloud_program() {
  if (cube.active()) {
    cloud_program = clipping_program;
    cloud_program->bind();
    auto clipping_planes = cube.package_planes();
    cloud_program->setUniformValueArray(planes_location, clipping_planes.data(),
                                        clipping_planes.size());

  } else {
    cloud_program = nonclipping_program;
    cloud_program->bind();
  }

  vertex_location = cloud_program->attributeLocation("vertex");
  color_location = cloud_program->attributeLocation("color");
}

void Widget::draw() {
  glEnable(GL_DEPTH_TEST);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, aa_width, aa_height);

  set_cloud_program();

#if 1
  if (!render_fbo || !render_fbo->isValid()) {
    QOpenGLFramebufferObjectFormat format;
    format.setAttachment(QOpenGLFramebufferObject::Depth);
    format.setSamples(4);
    render_fbo = std::unique_ptr<QOpenGLFramebufferObject>(
        new QOpenGLFramebufferObject(aa_width, aa_height, format));
    std::cout << "Allocated render_fbo" << std::endl;
  }

  if (!render_fbo->bind())
    std::cout << "Could not bind buffer!" << std::endl;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#endif
  cloud_program->setUniformValue("mvp_matrix", mvp);

  int idx = 0;
  long points_analyzed = 0;
  for (auto &v : vertex_buffers) {
    long num_points_to_draw = buffer_sizes[idx];

    v->bind();

    quintptr offset = 0;

    cloud_program->enableAttributeArray(vertex_location);
    cloud_program->setAttributeBuffer(vertex_location, GL_FLOAT, offset, 3,
                                      sizeof(VertexData));
    offset += 3 * sizeof(float);

    cloud_program->enableAttributeArray(color_location);
    cloud_program->setAttributeBuffer(color_location, GL_FLOAT, offset, 3,
                                      sizeof(VertexData));

    long true_num_points = std::min(
        num_points_to_draw, h_bins[binner(h_clipping_plane)] - points_analyzed);
    /*for (; true_num_points < num_points_to_draw &&
           cloud->at(true_num_points + points_analyzed).z < h_clipping_plane;
         ++true_num_points) {
    }*/

    if (true_num_points <= 0) {
      v->release();
      break;
    }

    glDrawArrays(GL_POINTS, 0, true_num_points);

    points_analyzed += num_points_to_draw;

    ++idx;
  }

  cloud_program->release();

#if 1

  if (!texture_fbo || !texture_fbo->isValid()) {
    texture_fbo = std::unique_ptr<QOpenGLFramebufferObject>(
        new QOpenGLFramebufferObject(aa_width, aa_height));
    std::cout << "Allocated texture_fbo" << std::endl;
  }

  QRect rect(0, 0, render_fbo->width(), render_fbo->height());
  QOpenGLFramebufferObject::blitFramebuffer(
      texture_fbo.get(), rect, render_fbo.get(), rect, GL_COLOR_BUFFER_BIT,
      GL_LINEAR, 0, 0,
      QOpenGLFramebufferObject::RestoreFramebufferBindingToDefault);

  glViewport(0, 0, aa_width / aa_factor, aa_height / aa_factor);
  aa_program->bind();
  glDisable(GL_DEPTH_TEST);

  aa_buffer->bind();
  quintptr offset = 0;
  aa_program->enableAttributeArray(position_location);
  aa_program->setAttributeBuffer(position_location, GL_FLOAT, offset, 2, 0);

  /*offset += 2 * sizeof(float);

  aa_program->enableAttributeArray(texcoord_location);
  aa_program->setAttributeBuffer(texcoord_location, GL_FLOAT, offset, 2,
                                 4 * sizeof(float));*/

  glBindTexture(GL_TEXTURE_2D, texture_fbo->texture());
  aa_program->setUniformValue(sampler_location, 0);
  aa_program->setUniformValue(
      viewport_location,
      QVector2D(aa_height / aa_factor, aa_width / aa_factor));

  glDrawArrays(GL_TRIANGLES, 0, 6);
  aa_program->release();
#endif
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

  static const Eigen::Array3d delta(5.5, 5.5, 5.5);

  max = average + delta * sigma / 2.0;
  min = average - delta * sigma / 2.0;

  std::cout << max << std::endl << std::endl << min << std::endl << std::endl;

  /*cloud->erase(std::remove_if(cloud->begin(), cloud->end(),
                              [&](PointType &p) {
                                bool in = true;
                                for (int i = 0; i < 3; ++i)
                                  if (p.getVector3fMap()[i] < min[i] ||
                                      p.getVector3fMap()[i] > max[i])
                                    in = false;

                                return !in;
                              }),
               cloud->end());*/

  double tmp = max[1];
  max[1] = max[2];
  max[2] = tmp;

  tmp = min[1];
  min[1] = min[2];
  min[2] = tmp;

  floor_box =
      Eigen::Vector2d(std::abs(max[0] - min[0]), std::abs(max[2] - min[2]));

  distance = std::max(
      10.0, std::min(50.0, std::sqrt((sigma * delta).square().sum()) / 2.0));
  start_distance = distance;

  camera_origin = Eigen::Vector2d(average[0], average[1]);
  camera_y = 0;

  Eigen::Vector3d start_eye =
      Eigen::Vector3d(camera_origin[0], 0, camera_origin[1]) +
      distance * look_vector_start;

  eye = Eigen::Vector2d(start_eye[0], start_eye[2]);
  eye_y = start_eye[1];
  rails_eye = eye;

  org_camera = camera_origin;
  org_eye = eye;

  h_clipping_plane = max[1];

  static const Eigen::Vector3d diag = Eigen::Vector3d(1, 1, 1).normalized();
  cube = ClippingCube(Eigen::Vector3d(eye[0], 0, eye[1]),
                      Eigen::Vector3d(eye[0], 0, eye[1]) + 15 * diag);
  cube.growX(35);
  cube.growZ(35);
  cube.growY(50);
  cube.rotate(PI / 4.0);
  cube.deactivate();
}

void Widget::timerEvent(QTimerEvent *) { update(); }

void Widget::filter() {

  if (std::abs(FLAGS_ss_factor - 0.0) < 1e-12)
    return;

  static std::random_device rng;
  static std::mt19937_64 gen(rng());
  static std::uniform_real_distribution<> dist(0.0, 1.0);

  auto tmp = cloud;
  cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
  for (auto &p : *tmp)
    if (dist(gen) >= FLAGS_ss_factor)
      cloud->push_back(p);

  /*pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(cloud);
  cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
  uniform_sampling.setRadiusSearch(1e-3);
  uniform_sampling.filter(*cloud);*/

  /* pcl::StatisticalOutlierRemoval<PointType> sor;
   sor.setInputCloud(cloud);
   sor.setMeanK(50);
   sor.setStddevMulThresh(2.0);
   cloud = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
   sor.filter(*cloud);*/

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