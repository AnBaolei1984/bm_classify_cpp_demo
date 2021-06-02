/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <signal.h>
#include <queue>
#include "Classify.hpp"
#include "utils.hpp"
#include <sys/time.h> 

namespace fs = boost::filesystem;
using namespace std;

queue <vector<cv::Mat> > g_mat_q;
pthread_mutex_t g_mutex;
int g_batch_size = 4;
int g_device_id = 0;
int g_proc_img_num = 0;
struct timeval g_start;

static void Classify(ClassifyNet &net, vector<cv::Mat>& images) {
  net.preForward(images);
  net.forward();
  vector<st_ClassifyResult> results = net.postForward();
#if 0
  for (size_t i = 0; i < results.size(); i++) {
    cout << " class id : " <<
       results[i].class_id << " score : " << results[i].score << endl;
  }
#endif
}

void video_main_proc(string bmodel_file) {
  ClassifyNet net(bmodel_file, g_device_id);
  g_batch_size = net.getBatchSize();
  struct timeval end;
  gettimeofday(&g_start, NULL);
  while(1) {
    if (g_mat_q.empty()) {
      continue;
    }
    pthread_mutex_lock(&g_mutex);
    auto batch_imgs = g_mat_q.front();
    g_mat_q.pop();
    pthread_mutex_unlock(&g_mutex);
    Classify(net, batch_imgs);
    g_proc_img_num += g_batch_size;
    gettimeofday(&end, NULL);
    double total_time =
       (end.tv_sec - g_start.tv_sec) * 1000000 + (end.tv_usec - g_start.tv_usec);
    total_time /= 1000000;
    cout << "=====Throughput is " <<
         g_proc_img_num / total_time << " image/s" << endl;
  }
}

int main(int argc, char **argv) {
  cout.setf(ios::fixed);
  if (argc < 3) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <video url>  <bmodel file> " << endl;
    exit(1);
  }

  string video_url = argv[1];
  string bmodel_file = argv[2];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  cv::VideoCapture cap;
  cap.open(video_url, 0, g_device_id);
  cap.set(cv::CAP_PROP_OUTPUT_YUV, 1);

  std::thread th_v(video_main_proc, bmodel_file);
  th_v.detach();
  size_t max_queue_num = 10;
  vector<cv::Mat> batch_imgs;
  while(1) {
    if (cap.isOpened()) {
      int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
      int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      cv::Mat img;
      cap >> img;
      if (img.rows != h || img.cols != w) {
        break;
      }
      batch_imgs.push_back(img);
      if ((int)batch_imgs.size() < g_batch_size) {
        continue;
      }
    } else {
      cout << "open video error" << endl;
      break;
    }
    pthread_mutex_lock(&g_mutex);
    if (g_mat_q.size() == max_queue_num) {
      g_mat_q.pop();
    }
    g_mat_q.push(batch_imgs);
    pthread_mutex_unlock(&g_mutex);
    batch_imgs.clear();
  }

  return 0;
}
