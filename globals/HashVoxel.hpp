#include <unordered_map>
#include <eigen3/Eigen/Eigen>


namespace std {
  template <>
  struct hash<Eigen::Vector3i>
  {
    std::size_t operator()(const Eigen::Vector3i & k) const {
      size_t seed = 0;
      for (int i = 0; i < k.size(); ++i) {
        seed ^= k[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace voxel {
	template<typename T>
	class HashVoxel {
		public:
			typedef T Scalar;
			HashVoxel() 
			{
			};
			bool insert(int i, int j, int k, const T & e) {
				Eigen::Vector3i key (i, j, k);
				auto it = map.find(key);
				if (it == map.cend()) {
					map.emplace(key, e);
					return true;
				} else
					return false;
			}
			T * operator()(int i, int j, int k) {
				Eigen::Vector3i key (i, j, k);
				auto it = map.find(key);
				if (it == map.cend())
					return NULL;
				else
					return &it->second;
			}


		private:
			std::unordered_map<Eigen::Vector3i, T> map;
	};
} // voxel	