#include "framerecorder.h"
using namespace std;

#include <iomanip>
#include <sstream>

FrameRecorder::FrameRecorder(const std::string &datapath_, bool save_)
    : datapath(datapath_), is_interrupted(false), max_frame_num(1000),
      frame_counter(0), _save{save_} {
  if (_save) {
    // start the worker thread here
    t = thread(&FrameRecorder::save, this);
    datapath.append("/record");
    // create the folder where all your saved image will be in
    QDir dir(QString::fromStdString(datapath));
    if (!dir.exists())
      dir.mkpath(QString::fromStdString(datapath));
  }
}

FrameRecorder::~FrameRecorder() {
  exit();
  if (t.joinable())
    t.join();
}

// the worker thread constantly take images and save
void FrameRecorder::save() {
  while (true) {
    unique_lock<mutex> lock(mt);
    cv.wait(lock, [&] {
      return this->saving_queue.size() >= 3 || this->is_interrupted.load();
    });
    if (is_interrupted.load()) {
      int total = (int)saving_queue.size() + frame_counter;
      while (saving_queue.size() >= 3) {
        auto img0 = saving_queue.front();
        saving_queue.pop_front();
        auto img1 = saving_queue.front();
        saving_queue.pop_front();
        auto img2 = saving_queue.front();
        saveFrame(img0, img1, img2);
        frame_counter++;
      }
      break;
    } else {
      auto img0 = saving_queue.front();
      saving_queue.pop_front();
      auto img1 = saving_queue.front();
      saving_queue.pop_front();
      auto img2 = saving_queue.front();
      lock.unlock();
      saveFrame(img0, img1, img2);
      frame_counter++;
    }
  }
}

bool FrameRecorder::submit_frame(cv::Mat img) {
  if (!_save)
    return true;
  lock_guard<mutex> guard(mt);
  saving_queue.push_back(img);

  if (saving_queue.size() >= 3)
    cv.notify_all();

  return true;
}

void FrameRecorder::saveFrame(cv::Mat &img0, cv::Mat &img1, cv::Mat &img2) {
  if (!last_img.data)
    last_img = img0;
  cv::Mat output;
  cv::addWeighted(last_img, 0.1, img0, 0.4, 0.0, output);
  cv::addWeighted(output.clone(), 1.0, img1, 0.4, 0.0, output);
  cv::addWeighted(output.clone(), 1.0, img2, 0.1, 0.0, output);
  std::ostringstream ss;
  ss << datapath << "/img" << std::setfill('0') << std::setw(6) << frame_counter
     << ".png";
  cv::imwrite(ss.str(), output);
  last_img = img1;
}
