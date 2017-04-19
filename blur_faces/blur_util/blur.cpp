#include <etw_utils.hpp>
#include <omp.h>
#include <opencv/cv.hpp>
#include <scan_gflags.h>

fs::path release_folder("/home/erik/Projects/ReleaseData/");
fs::path base_folder("/home/erik/Projects/c++/blur_faces/");

using Rect = std::tuple<double, double, double, double>;

cv::Vec3d apply_blur(const cv::Mat3b &img, int x, int y, double sigma, int KX,
                     int KY) {
  assert(KX % 2 == 1 && "KX must be odd!");
  assert(KY % 2 == 1 && "KY must be odd!");
  cv::Vec3d res(0, 0, 0);
  double total_weight = 0;
  for (int j = -KY / 2; j < KY / 2; ++j) {
    const int yprime = y + j < 0 || y + j >= img.rows ? y - j : y + j;
    for (int i = -KX / 2; i < KX / 2; ++i) {
      const int xprime = x + i < 0 || x + i >= img.cols ? y - i : x + i;
      const double weight = std::exp(
          -((xprime - x) * (xprime - x) + (yprime - y) * (yprime - y)) /
          (2 * sigma * sigma));
      res += img(yprime, xprime) * weight;
      total_weight += weight;
    }
  }
  res /= total_weight;
  return res;
}

cv::Mat3b blur_img(const cv::Mat3b &img, const std::vector<Rect> &rects) {
  auto res = img.clone();
  for (auto && [ x, y, width, height ] : rects) {
    int ix = x * img.cols;
    int iy = y * img.rows;
    int iwidth = width * img.cols;
    int iheight = height * img.rows;

    int x_mid = ix + iwidth / 2;
    int y_mid = iy + iheight / 2;
    for (int j = 0; j < iheight; ++j) {
      for (int i = 0; i < iwidth; ++i) {
        const int xprime = ix + i;
        const int yprime = iy + j;
        constexpr double SCALING_FACTOR = 6.0;
        const int KY =
            static_cast<int>(13 +
                             (iheight / 2 - std::abs(yprime - y_mid)) /
                                 SCALING_FACTOR) |
            0x1;

        const int KX =
            static_cast<int>(
                13 + (iwidth / 2 - std::abs(xprime - x_mid)) / SCALING_FACTOR) |
            0x1;
        const double sigma =
            4.0 * std::exp(1) *
            std::exp(-(xprime - x_mid) * (xprime - x_mid) / (iwidth * iwidth) -
                     (yprime - y_mid) * (yprime - y_mid) / (iheight * iheight));
        res(yprime, xprime) = apply_blur(img, xprime, yprime, sigma, KX, KY);
      }
    }
  }
  return res;
}

int main(int argc, char const *argv[]) {
  fs::path rects_folder("/home/erik/Projects/c++/blur_faces/rects/");
  for (auto &data_folder : utils::parse_folder(rects_folder)) {
    const auto floor_id =
        data_folder.string().substr(data_folder.string().rfind("/") + 1, 4);
    fs::path ptx_folder = base_folder / floor_id / "PTXFiles/";
    fmt::print("{}\n", ptx_folder);

    fs::path release_ptx_folder = release_folder / floor_id / "PTX/";
    fs::create_directories(release_ptx_folder);

    fs::path release_panorama_folder = release_folder / floor_id / "imgs/";
    fs::create_directories(release_panorama_folder);

    auto files = utils::parse_folder(ptx_folder);
#pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int i = 0; i < files.size(); ++i) {
      auto &ptx_file = files[i];
      const auto[building_name, scan_id] = parse_name(ptx_file);

      fs::path release_ptx_file =
          release_ptx_folder / "{}_scan_{}.ptx"_format(building_name, scan_id);

      fs::path release_img_file =
          release_panorama_folder /
          "{}_pano_{}.png"_format(building_name, scan_id);

      fs::path rects_file = rects_folder / floor_id /
                            "{}_rects_{}.txt"_format(building_name, scan_id);

      if (false && fs::exists(release_ptx_file) && fs::exists(release_img_file))
        continue;

      std::vector<Rect> rects;
      if (fs::exists(rects_file)) {
        std::ifstream in(rects_file.string(), std::ios::in);
        double x, y, width, height;
        while (in >> x >> y >> width >> height)
          rects.emplace_back(x, y, width, height);
      }

      fmt::print("{}\n", rects_file);

      int rows, cols;
      std::vector<scan::PointXYZRGBA> points;
      {
        fs::path binary_file =
            base_folder / floor_id / "binaryFiles" /
            "{}_binary_{}.dat"_format(building_name, scan_id);
        std::ifstream in(binary_file.string(), std::ios::in | std::ios::binary);
        in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        points.resize(rows * cols);
        for (auto &p : points)
          p.loadFromFile(in);
      }

      cv::Mat3b pano(rows, cols);
      std::vector<std::tuple<size_t, size_t, size_t>> img_cloud;
      int row = rows - 1;
      int col = 0.995 * (cols - 1) / 2.0;
      int idx = 0;
      for (auto &element : points) {
        const int real_row = row;
        const int real_col = cols - col - 1;
        pano(real_row, real_col)[0] = element.rgb[2];
        pano(real_row, real_col)[1] = element.rgb[1];
        pano(real_row, real_col)[2] = element.rgb[0];

        img_cloud.emplace_back(real_row, real_col, idx);
        ++idx;

        if (row == 0) {
          row = rows - 1;
          col = col == 0 ? cols - 1 : col - 1;
        } else
          --row;
      }
      auto blurred = blur_img(pano, rects);

      cv::imwrite(release_img_file.string(), blurred);

      for (auto && [ row, col, idx ] : img_cloud) {
        points[idx].rgb[0] = blurred(row, col)[2];
        points[idx].rgb[1] = blurred(row, col)[1];
        points[idx].rgb[2] = blurred(row, col)[0];
      }

      std::ofstream release_out(release_ptx_file.string());
      {
        /* Copy over the header from the old PTX_file */
        std::ifstream ptx_in(ptx_file.string(), std::ios::in);
        std::string tmp;
        for (int i = 0; i < 10; ++i) {
          std::getline(ptx_in, tmp);
          fmt::print(release_out, "{}\n", tmp);
        }
      }
      for (auto &p : points)
        fmt::print(release_out, "{:f} {:f} {:f} {:f} {:d} {:d} {:d}\n",
                   p.point[0], p.point[1], p.point[2], p.intensity, p.rgb[0],
                   p.rgb[1], p.rgb[2]);
    }
  }

  return 0;
}