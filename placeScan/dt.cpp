#include "dt.hpp"
#include <math.h>

static constexpr double INF = 1e20;

Eigen::MatrixXd dt::distanceTransform(const Eigen::MatrixXd &image) {
  int width = image.cols();
  int height = image.rows();

  Eigen::MatrixXd out(height, width);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (image(y, x) > 0)
        out(y, x) = 0;
      else
        out(y, x) = INF;
    }
  }

  dt(out);
  double *outPtr = out.data();
  double maxV = 0;
  for (int i = 0; i < out.size(); ++i) {
    *(outPtr + i) = sqrt(*(outPtr + i));
    maxV = std::max(maxV, *(outPtr + i));
  }
  for (int i = 0; i < out.size(); ++i) {
    *(outPtr + i) = *(outPtr + i) / maxV;
  }

  return out;
}

/* dt of 2d function using squared distance */
void dt::dt(Eigen::MatrixXd &image) {
  int width = image.cols();
  int height = image.rows();
  Eigen::VectorXf f(std::max(width, height));

  // transform along columns
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      f[y] = image(y, x);
    }
    Eigen::VectorXf d = dt::dt(f);
    for (int y = 0; y < height; y++) {
      image(y, x) = d[y];
    }
  }

  // transform along rows
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      f[x] = image(y, x);
    }
    Eigen::VectorXf d = dt(f);
    for (int x = 0; x < width; x++) {
      image(y, x) = d[x];
    }
  }
}

/* dt of 1d function using squared distance */
Eigen::VectorXf dt::dt(Eigen::VectorXf &f) {
  int n = f.size();
  Eigen::VectorXf d(n);
  Eigen::VectorXf v(n);
  Eigen::VectorXf z(n + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (int q = 1; q <= n - 1; q++) {
    float s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    if (s <= z[k])
      --k;
    else {
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = +INF;
    }
  }

  k = 0;
  for (int q = 0; q <= n - 1; q++) {
    while (z[k + 1] < q)
      k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }

  return d;
}