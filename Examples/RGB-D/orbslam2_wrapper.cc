#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "System.h"
#include "Map.h"
#include "Unity.h"

#include "glog/logging.h"

#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"

namespace py = pybind11;
namespace fs = std::experimental::filesystem;

static std::string dataset_path = "";
static std::string associations_path = "";
static std::string output_result_path = "";
static ORB_SLAM2::System* slam = nullptr;

void DatasetPath(const std::string& data_path, const std::string& assoc_path,
                 const std::string& output_path) {
  dataset_path = data_path;
  associations_path = assoc_path;
  output_result_path = output_path;
  LOG(INFO) << "dataset_path: " << dataset_path << ", " << associations_path;
}

void LoadImages(const std::string& strAssociationFilename,
                std::vector<std::vector<std::string>>& vstrImageFilenames,
                std::vector<double>& vTimestamps) {
  std::ifstream fAssociation(strAssociationFilename);
  std::string line;

  while (std::getline(fAssociation, line)) {
    std::stringstream ss(line);
    std::string value;
    int col_idx = 0;
    bool init = false;

    while (ss >> value) {
      if (!init) {
        init = true;
        vTimestamps.push_back(std::stod(value));
        continue;
      }
      if (col_idx >= vstrImageFilenames.size()) {
        vstrImageFilenames.push_back(std::vector<std::string>());
      }
      vstrImageFilenames[col_idx].push_back(value);
      col_idx++;
    }
  }
}

std::unordered_map<std::string, cv::Mat> MaskImages(
    int col, const std::vector<std::vector<std::string>>& img_strs, const std::string& base_path) {
  std::unordered_map<std::string, cv::Mat> imMasks;
  for (size_t nj = 3; nj < img_strs.size(); ++nj) {
    if (img_strs[nj][col] == "0") continue;
    cv::Mat imMask = cv::imread(base_path + "/" + img_strs[nj][col], cv::IMREAD_GRAYSCALE);
    size_t pos = img_strs[nj][col].find('/');
    if (pos == std::string::npos || imMask.empty()) continue;
    std::string directory = img_strs[nj][col].substr(0, pos);
    imMasks.insert({directory, imMask});
  }
  return imMasks;
}

std::string OutputResult() {
  std::string output_folder = output_result_path;
  if (output_folder.back() != '/') output_folder += "/";
  fs::create_directories(output_folder);
  return output_folder;
}

void RunSlam() {
  py::gil_scoped_release release;

  std::vector<double> timestamps;
  std::vector<std::vector<std::string>> image_strs;
  LoadImages(associations_path, image_strs, timestamps);

  std::vector<std::string> rgb_strs = image_strs[0];
  std::vector<std::string> depth_strs = image_strs[1];
  std::vector<std::string> segment_strs = image_strs[2];

  LOG_IF(ERROR, rgb_strs.empty()) << "No images found in provided path.";
  LOG_IF(ERROR, rgb_strs.size() != depth_strs.size())
      << "Different number of images for rgb and depth.";

  std::vector<float> vTimesTrack(rgb_strs.size());

  LOG(INFO) << "Start processing sequence with " << rgb_strs.size() << " images";

  for (int ni = 0; ni < rgb_strs.size(); ni++) {
    std::string rgb_path = dataset_path + "/" + rgb_strs[ni];
    std::string depth_path = dataset_path + "/" + depth_strs[ni];
    std::string segment_path = dataset_path + "/" + segment_strs[ni];

    double tframe = timestamps[ni];
    cv::Mat imRGB = cv::imread(rgb_path, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imD = cv::imread(depth_path, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat imSegment = cv::imread(segment_path, cv::IMREAD_COLOR);
    auto imMasks = MaskImages(ni, image_strs, dataset_path);

    if (imRGB.empty() || imD.empty()) {
      LOG(ERROR) << "Failed to load image at: " << dataset_path;
      continue;
    }

    auto t1 = std::chrono::steady_clock::now();
    slam->TrackRGBD(imRGB, imD, imSegment, imMasks, rgb_path, tframe);
    auto t2 = std::chrono::steady_clock::now();

    vTimesTrack[ni] = std::chrono::duration<float>(t2 - t1).count();
    usleep(2e4);
  }

  std::sort(vTimesTrack.begin(), vTimesTrack.end());
  float totaltime = std::accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
  LOG(INFO) << "Median time: " << vTimesTrack[vTimesTrack.size() / 2];
  LOG(INFO) << "Mean time: " << totaltime / vTimesTrack.size();

  slam->SaveTrajectoryTUM(output_result_path + "CameraTrajectory.txt");
  slam->SaveKeyFrameTrajectoryTUM(output_result_path + "KeyFrameTrajectory.txt");
  slam->SaveMapObjectsPly(output_result_path);
  slam->SaveMapObjectsOBJ(output_result_path + "map_objects_supersquadic.ply");
  slam->Shutdown();
}

void init_slam(const std::string& vocab_path, const std::string& config_path) {
  if (slam) delete slam;
  slam = new ORB_SLAM2::System(vocab_path, config_path, ORB_SLAM2::System::RGBD, true);
  LOG(INFO) << "--- init slam buffer ---";
}

ORB_SLAM2::System* get_slam_system() {
  return slam;
}

void track_image(py::array_t<uint8_t> image_np, double timestamp) {
  if (!slam) return;
  py::buffer_info buf = image_np.request();
  if (buf.ndim != 2) throw std::runtime_error("Expected 2D grayscale image");

  cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC1, (uint8_t*)buf.ptr);
  slam->TrackMonocular(img.clone(), timestamp);
}

void shutdown_slam() {
  if (slam) {
    slam->Shutdown();
    delete slam;
    slam = nullptr;
  }
}

py::array OccupancyMap() {
  cv::Mat grid_map = slam->GridMap();
  std::vector<std::size_t> shape;
  std::vector<std::size_t> strides;

  if (grid_map.channels() == 1) {
    shape = {(size_t)grid_map.rows, (size_t)grid_map.cols};
    strides = {(size_t)grid_map.step[0], (size_t)grid_map.elemSize()};
  } else {
    shape = {(size_t)grid_map.rows, (size_t)grid_map.cols, (size_t)grid_map.channels()};
    strides = {(size_t)grid_map.step[0], (size_t)grid_map.elemSize1() * grid_map.channels(), (size_t)grid_map.elemSize1()};
  }

  return py::array(py::buffer_info(
      grid_map.data,                      // 指针
      grid_map.elemSize1(),              // 每个元素大小 (通常是1或4)
      py::format_descriptor<uchar>::format(),  // 格式 (uint8)
      shape.size(),                 // 维度
      shape,                        // 形状
      strides                       // 步长
  ));
}

py::array_t<float> GlobalMap() {
  size_t num_points = slam->GlobalMap()->size();
  if (num_points == 0) {
    LOG(ERROR) << "Global map is empty";
    return py::array_t<float>();
  }

  py::array_t<float> result = py::array_t<float>(num_points * 7);
  result.resize({ssize_t(num_points), ssize_t(7)});
  auto buf = result.mutable_unchecked<2>();

  for (size_t i = 0; i < num_points; ++i) {
    const auto& pt = slam->GlobalMap()->points[i];
    buf(i, 0) = pt.x;
    buf(i, 1) = pt.y;
    buf(i, 2) = pt.z;
    buf(i, 3) = static_cast<float>(pt.r / 255.0f);  // 可选：保留 [0,1] 范围
    buf(i, 4) = static_cast<float>(pt.g / 255.0f);
    buf(i, 5) = static_cast<float>(pt.b / 255.0f);
    buf(i, 6) = pt.a / 255.0f;
  }
  return result;
}

PYBIND11_MODULE(orbslam2_py, m) {
  m.def("init_slam", &init_slam, "Initialize ORB-SLAM2");
  m.def("get_slam_system", &get_slam_system, py::return_value_policy::reference, "Get SLAM system");
  m.def("shutdown_slam", &shutdown_slam, "Shutdown SLAM system");
  m.def("track_image", &track_image, "Track one grayscale image");
  m.def("RunSlam", &RunSlam, "Run the full SLAM pipeline");
  m.def("DatasetPath", &DatasetPath, "Set dataset and output paths");
  m.def("LoadImages", &LoadImages, "Load image paths and timestamps");
  m.def("GlobalMap", &GlobalMap, "Global map");
  m.def("OccupancyMap", &OccupancyMap, "Occupancy Grid map");

  py::class_<ORB_SLAM2::System>(m, "System")
      .def("TrackRGBD", &ORB_SLAM2::System::TrackRGBD)
      .def("TrackMonocular", &ORB_SLAM2::System::TrackMonocular)
      .def("Shutdown", &ORB_SLAM2::System::Shutdown)
      .def("SaveTrajectoryTUM", &ORB_SLAM2::System::SaveTrajectoryTUM)
      .def("SaveKeyFrameTrajectoryTUM", &ORB_SLAM2::System::SaveKeyFrameTrajectoryTUM)
      .def("SaveMapObjectsPly", &ORB_SLAM2::System::SaveMapObjectsPly)
      .def("SaveMapObjectsOBJ", &ORB_SLAM2::System::SaveMapObjectsOBJ);
}