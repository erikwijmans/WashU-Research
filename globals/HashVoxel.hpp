#pragma once
#ifndef HASH_VOXEL_HPP
#define HASH_VOXEL_HPP

#include <unordered_map>
#include <eigen3/Eigen/Eigen>
#include <memory>

namespace std {
  template<>
  template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  struct hash<Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > >
  {
    std::size_t operator()(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > & k) const {
      size_t seed = 0;
      auto dataPtr = k.data();
      for (int i = 0; i < k.size(); ++i) {
        seed ^= *(dataPtr + i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
  };
} // voxel


#endif // HASH_VOXEL_HPP