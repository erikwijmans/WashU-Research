#pragma once
#ifndef DIRECT_VOXEL_HPP
#define DIRECT_VOXEL_HPP

#include <eigen3/Eigen/Eigen>
#include <memory>
#include <scan_typedefs.hpp>
#include <vector>

namespace voxel {
template <typename V> class DirectVoxel {
public:
  typedef Eigen::Vector3i K;
  typedef std::shared_ptr<DirectVoxel<V>> Ptr;
  typedef const std::shared_ptr<DirectVoxel<V>> ConstPtr;
  typedef Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
  typedef std::vector<Mat> Mem;

  DirectVoxel(K &min, K &max)
      : _min{min}, _max{max}, x{max[0] - min[0]}, y{max[1] - min[1]},
        z{max[2] - min[2]}, mem{z, Mat::Zero(y, x)} {};
  DirectVoxel(K &&min, K &&max)
      : _min{min}, _max{max}, x{max[0] - min[0]}, y{max[1] - min[1]},
        z{max[2] - min[2]}, mem{z, Mat::Zero(y, x)} {};

  DirectVoxel(int x, int y, int z) : DirectVoxel(K::Zero(), K(x, y, z)){};

  template <typename... Targs> static inline Ptr Create(Targs &... args) {
    return std::make_shared<DirectVoxel<V>>(std::forward<Targs>(args)...);
  };
  template <typename... Targs> static inline Ptr Create(Targs &&... args) {
    return std::make_shared<DirectVoxel<V>>(std::forward<Targs>(args)...);
  };

  V &operator()(const K &key) {
    K index = key - min();
    checkBounds(index);

    return mem[index[2]](index[1], index[0]);
  }

  V &operator()(int x, int y, int z) {
    K key(x, y, z);
    return operator()(key);
  };
  V &at(int x, int y, int z) { return operator()(x, y, z); };
  V &at(const K &key) { return operator()(key); };

  inline K &max() { return _max; };
  inline K &min() { return _min; };

  template <typename PairFunc>
  void update(const DirectVoxel &o, PairFunc updateRule) {
    assert(min() == o.min() && max() == o.max() && "Grids not the same size!");
    for (int k = 0; k < z; ++k)
      for (int j = 0; j < y; ++j)
        for (int i = 0; i < y; ++i)
          mem[k](j, i) = updateRule(mem[k](j, i), o.mem[k](j, i));
  }

  void operator+=(const DirectVoxel &o) {
    update(o, [](V &v1, V &v2) { return v1 + v2; });
  }

private:
  K _min, _max;
  size_t x, y, z;
  Mem mem;

  void checkBounds(const K &key) {
    auto minPtr = min().data();
    auto maxPtr = max().data();
    auto keyPtr = key.data();
    for (int i = 0; i < key.size(); ++i) {
      assert(*(keyPtr + i) >= *(minPtr + i) && "Index out of bounds!");
      assert(*(keyPtr + i) < *(maxPtr + i) && "Index out of bounds!");
    }
  };
};
} // voxel

#endif // DIRECT_VOXEL_HPP