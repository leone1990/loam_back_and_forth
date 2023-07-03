#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct CloudPointIndexIdx {
  unsigned int idx;
  unsigned int cloud_point_index;

  CloudPointIndexIdx(const unsigned int idx_,
                     const unsigned int _cloud_point_index)
      : idx(idx_), cloud_point_index(_cloud_point_index) {}
  bool operator<(const CloudPointIndexIdx &p) const { return (idx < p.idx); }
};

// 这个体素滤波是修改自pcl::VoxelGrid
template <typename PointT>
class VoxelFilter {
 public:
  // 构造的时候直接输入所需点云
  explicit VoxelFilter(typename pcl::PointCloud<PointT>::Ptr input)
      : input_(input) {
    indices_.reserve(input_->size());
    for (size_t i = 0; i < input->size(); ++i) {
      indices_.emplace_back(static_cast<int>(i));
    }
  };

  void ApplyFilter(pcl::PointCloud<PointT> &output);

  inline void SetLeafSize(const float ls) { SetLeafSize(ls, ls, ls); }

  inline void SetLeafSize(const float lx, const float ly, const float lz) {
    leaf_size_[0] = lx;
    leaf_size_[1] = ly;
    leaf_size_[2] = lz;
    // Avoid division errors
    if (leaf_size_[3] == 0) {
      leaf_size_[3] = 1;
    }
    // Use multiplications instead of divisions
    inverse_leaf_size_ = Eigen::Array4f::Ones() / leaf_size_.array();
  }

 private:
  void GetMinMax3D(const pcl::PointCloud<PointT> &cloud,
                   const std::vector<int> &indices, Eigen::Vector4f &min_pt,
                   Eigen::Vector4f &max_pt);

 private:
  typename pcl::PointCloud<PointT>::Ptr input_ = nullptr;
  std::vector<int> indices_;

  // The size of a leaf.
  Eigen::Vector4f leaf_size_;

  // Internal leaf sizes stored as 1/leaf_size_ for efficiency reasons.
  Eigen::Array4f inverse_leaf_size_;

  Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;

  unsigned int min_points_per_voxel_ = 0;
};