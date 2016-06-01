/**
  Implements a hashmap based voxelgrid.  This has the benefit of
  being significantly less memory intensive than a direct-mapped
  voxel grid, however, it has the disadvantage of being slightly slower.

  This classes uses awful looking variardic templating so that it is
  really easy to interact with:

    HashVoxel<Eigen::Vector3i, double> myVoxel
    myVoxel.insert(value, x, y, z)
    value == *myVoxel(x, y, z)

  The K (key) must be a Eigen Matrix type.  If you know how many
  dimensions the key will be, something like Eigen::Vector2i is
  preferable, otherwise Eigen::VectorXi will work.
*/

#pragma once
#ifndef HASH_VOXEL_HPP
#define HASH_VOXEL_HPP

#include <unordered_map>
#include <eigen3/Eigen/Eigen>
#include <memory>

// Defines a hasher that is capable of hasing an arbitrary Eigen Matrix
namespace std {
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  struct hash<Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > >
  {
    //NB:: More than likely, indicies into the HashVoxel will be in
    // the range [0, 10000] which doesn't satisfy the uniform hashing
    // assumption very well, so A is used to spread those out.
    static constexpr double A = 1.6180339887498948482*1e5;
    hash<double> h;
    size_t operator()(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > & k) const {
      size_t seed = 0;
      auto dataPtr = k.data();
      for (int i = 0; i < k.size(); ++i) {
        seed ^= h(*(dataPtr + i)*A) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace voxel {
  template<typename K, typename V>
  class HashVoxel {
    public:
      typedef std::shared_ptr<V> VPtr;
      typedef std::shared_ptr<HashVoxel<K, V> > Ptr;
      HashVoxel(K & min, K & max) :
        _min {min}, _max {max}
      {
      };
      HashVoxel(K && min, K && max) :
        _min {min}, _max {max}
      {
      };
      template<typename... Targs>
      static inline Ptr Create(Targs... args) {
        return std::make_shared<HashVoxel<K, V> >(std::forward<Targs>(args)...);
      }
      template<typename... Kargs>
      VPtr insert(VPtr v, Kargs... args) {
        K key (std::forward<Kargs>(args)...);
        checkBounds(key);
        auto it = map.find(key);
        if (it == map.cend()) {
          map.emplace(key, v);
          return v;
        } else
          return nullptr;
      };
      template<typename... Kargs>
      VPtr insert(V & v, Kargs... args) {
        return insert(std::make_shared<V>(v), std::forward<Kargs>(args)...);
      };
      template<typename... Kargs>
      VPtr insert(V && v, Kargs... args) {
        return insert(std::make_shared<V>(v), std::forward<Kargs>(args)...);
      };
      template<typename... Kargs>
      VPtr operator()(Kargs... args) {
        K key (std::forward<Kargs>(args)...);
        checkBounds(key);
        auto it = map.find(key);
        if (it == map.cend())
          return nullptr;
        else
          return it->second;
      };
      template<typename... Kargs>
      VPtr at(Kargs... args) {
        return operator()(std::forward<Kargs>(args)...);
      }

      inline K & max() { return _max; };
      inline K & min() {return _min; };
    private:
      std::unordered_map<K, VPtr> map;
      K _min, _max;
      void checkBounds(K key) {
        auto minPtr = min().data();
        auto maxPtr = max().data();
        auto keyPtr = key.data();
        for (int i = 0; i < key.size(); ++i) {
          assert(*(keyPtr + i) >= *(minPtr + i));
          assert(*(keyPtr + i) < *(maxPtr + i));
        }
      }
      HashVoxel();
  };
} // voxel


#endif // HASH_VOXEL_HPP