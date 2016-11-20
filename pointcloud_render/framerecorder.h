#ifndef FRAMERECORDER_H
#define FRAMERECORDER_H

#include <QDir>
#include <QImage>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv/cv.hpp>
#include <string>
#include <thread>

/**
 * Usage: Once you grab the framebuffer into a QImage, you call
 * FrameRecorder::submitFrame().
 * The newly arrived frames will be first pushed into a queue, and a worder
 * thread will constantly save images in
 * that queue into the disk.
 * There is a maximum limit of queue size, if the queue is full, the
 * submitFrame() will return false, then you might
 * want to pause the rendering to wait the worker thread
 */

class FrameRecorder {
public:
  FrameRecorder(const std::string &datapath_);
  ~FrameRecorder();
  bool submit_frame(cv::Mat img);
  inline void exit() {
    is_interrupted.store(true);
    cv.notify_all();
  }

private:
  void save();
  void saveImage(cv::Mat &img);
  std::string datapath;
  std::deque<cv::Mat> saving_queue;
  std::condition_variable cv;
  std::mutex mt;
  std::thread t;
  std::atomic<bool> is_interrupted;
  const int max_frame_num;
  int frame_counter;
};

#endif // FRAMERECORDER_H
