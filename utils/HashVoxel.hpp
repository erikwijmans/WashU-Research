#pragma once
#ifndef HASH_VOXEL_HPP
#define HASH_VOXEL_HPP

#include <eigen3/Eigen/Eigen>
#include <etw_utils.hpp>
#include <memory>
#include <unordered_map>

namespace voxel {
template <typename K, typename V> class HashVoxel {
public:
  typedef std::shared_ptr<V> VPtr;
  typedef std::shared_ptr<HashVoxel<K, V>> Ptr;
  typedef const std::shared_ptr<HashVoxel<K, V>> ConstPtr;
  typedef std::unordered_map<K, VPtr> Map;

  HashVoxel(K &min, K &max) : _min{min}, _max{max}, check{true} {};
  HashVoxel(K &&min, K &&max) : _min{min}, _max{max}, check{true} {};
  HashVoxel() : check{false} {
    auto minPtr = min().data();
    auto maxPtr = max().data();
    for (int i = 0; i < max().size(); ++i) {
      minPtr[i] = 1e8;
      maxPtr[i] = -1e8;
    }
  };

  template <typename... Targs> static inline Ptr Create(Targs &... args) {
    return std::make_shared<HashVoxel<K, V>>(std::forward<Targs>(args)...);
  };
  template <typename... Targs> static inline Ptr Create(Targs &&... args) {
    return std::make_shared<HashVoxel<K, V>>(std::forward<Targs>(args)...);
  };

  VPtr insert(VPtr v, K key) {
    if (check)
      checkBounds(key);
    else
      update(key);
    auto it = map.find(key);
    if (it == map.cend()) {
      map.emplace(key, v);
      return v;
    } else
      return nullptr;
  }

  template <typename... Kargs> VPtr insert(VPtr v, Kargs... args) {
    K key(std::forward<Kargs>(args)...);
    return insert(v, key);
  };

  template <typename... Kargs> VPtr insert(const V &v, Kargs... args) {
    return insert(std::make_shared<V>(v), std::forward<Kargs>(args)...);
  };
  template <typename... Kargs> VPtr insert(const V &&v, Kargs... args) {
    return insert(std::make_shared<V>(v), std::forward<Kargs>(args)...);
  };

  VPtr operator()(K &key) {
    assert(checkBounds(key) && "Not in bounds!");
    auto it = map.find(key);
    if (it == map.cend())
      return nullptr;
    else
      return it->second;
  }

  template <typename... Kargs> VPtr operator()(Kargs... args) {
    K key(std::forward<Kargs>(args)...);
    return operator()(key);
  };
  template <typename... Kargs> VPtr at(Kargs... args) {
    return operator()(std::forward<Kargs>(args)...);
  };

  inline K &max() { return _max; };
  inline K &min() { return _min; };

  typename Map::const_iterator begin() const { return map.cbegin(); };
  typename Map::const_iterator end() const { return map.cend(); };

  template <typename PairFunc>
  void update(const HashVoxel &o, PairFunc updateRule) {
    for (auto &pair : o) {
      K key;
      VPtr ov;
      std::tie(key, ov) = pair;

      auto v = operator()(key);
      if (v)
        *v = updateRule(*v, *ov);
      else
        insert(ov, key);
    }
  }

  void operator+=(const HashVoxel &o) {
    update(o, [](V &v1, V &v2) { return v1 + v2; });
  }

  bool checkBounds(const K &key) {
    if (check) {
      auto minPtr = min().data();
      auto maxPtr = max().data();
      auto keyPtr = key.data();
      for (int i = 0; i < key.size(); ++i) {
        if (*(keyPtr + i) < *(minPtr + i) || *(keyPtr + i) >= *(maxPtr + i))
          return false;
      }
    }

    return true;
  };

private:
  Map map;
  K _min, _max;
  const bool check;

  void update(const K &key) {
    auto minPtr = min().data();
    auto maxPtr = max().data();
    auto keyPtr = key.data();
    for (int i = 0; i < key.size(); ++i) {
      minPtr[i] = std::min(keyPtr[i], minPtr[i]);
      maxPtr[i] = std::max(keyPtr[i], maxPtr[i]);
    }
  }
};
} // voxel

#endif // HASH_VOXEL_HPP