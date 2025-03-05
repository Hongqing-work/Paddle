// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/distributed/collective/deep_ep/include/event_pool.h"
#include "glog/logging.h"

namespace deep_ep::detail {

EventPool &EventPool::instance() {
  static EventPool pool;
  return pool;
}

EventPool::~EventPool() {
  const auto &DestroyEvent = [](cudaEvent_t *event) {
    cudaError_t e = cudaEventDestroy(*event);
    if (e != cudaSuccess) {
      LOG(FATAL) << "CUDA event destroy failed: " << cudaGetErrorString(e);
    }
  };
  const auto &CheckComplishAndDestroy = [&](cudaEvent_t *event) -> bool {
    if (cudaEventQuery(*event) == cudaSuccess) {
      DestroyEvent(event);
      return true;
    }
    if (cudaEventDestroy(*event) == cudaErrorNotReady) {
      LOG(FATAL) << "event is not completed or when destroying event pool.";
      return false;
    }
    LOG(FATAL) << "failed on cudaEventQuery when destroying event pool.";
    return false;
  };
  while (!incomplished_events_.empty()) {
    cudaEvent_t *event = &(incomplished_events_.back());
    if (!CheckComplishAndDestroy(event)) {
      LOG(FATAL) << "failed on destroying event when destroying event pool.";
    }
    incomplished_events_.pop_back();
  }
}

cudaEvent_t *EventPool::CreateCudaEventFromPool() {
  std::lock_guard<std::mutex> guard(mtx_);

  const auto &CreateNewEvent = [&]() -> cudaEvent_t * {
    cudaEvent_t new_event;
    CUDA_CHECK(cudaEventCreate(&new_event));
    incomplished_events_.push_back(new_event);
    return &incomplished_events_.back();
  };

  const auto &CreateNewOrReuseEvent = [&]() -> cudaEvent_t * {
    cudaEvent_t *event = &(incomplished_events_.front());
    if (cudaEventQuery(*event) == cudaSuccess) {
      incomplished_events_.pop_front();
      incomplished_events_.push_back(*event);
      return event;
    }
    return CreateNewEvent();
  };

  if (incomplished_events_.empty()) {
    return CreateNewEvent();
  }
  return CreateNewOrReuseEvent();
}
}  // namespace deep_ep::detail
