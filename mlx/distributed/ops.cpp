// Copyright © 2024 Apple Inc.

#include <mutex>
#include <sstream>
#include <unordered_map>

#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

namespace {

Group to_group(std::optional<Group> group) {
  if (group.has_value()) {
    return group.value();
  } else {
    return distributed::init();
  }
}

// Send and Recv have no natural data dependency on each other, so independent
// siblings on the same comm stream can be reordered by eval. We chain each
// new Send/Recv to the previous one on the same (group, stream) as a hidden
// input, forcing eval to dispatch them in construction order.
struct CommOrderKey {
  void* group_ptr;
  int stream_index;
  bool operator==(const CommOrderKey& o) const {
    return group_ptr == o.group_ptr && stream_index == o.stream_index;
  }
};

struct CommOrderKeyHash {
  size_t operator()(const CommOrderKey& k) const {
    return std::hash<void*>()(k.group_ptr) ^
        (std::hash<int>()(k.stream_index) << 1);
  }
};

std::mutex& comm_order_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<CommOrderKey, array, CommOrderKeyHash>&
comm_order_registry() {
  static std::unordered_map<CommOrderKey, array, CommOrderKeyHash> r;
  return r;
}

template <class BuildFn>
array make_chained(
    const Group& group,
    Stream stream,
    std::vector<array> inputs,
    BuildFn&& build) {
  CommOrderKey key{group.raw_group().get(), stream.index};
  std::lock_guard<std::mutex> lock(comm_order_mutex());
  auto& reg = comm_order_registry();
  auto it = reg.find(key);
  if (it != reg.end()) {
    inputs.push_back(it->second);
  }
  array out = build(std::move(inputs));
  reg.insert_or_assign(key, out);
  return out;
}

} // namespace

array all_sum(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Sum),
      {x});
}

array all_max(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Max),
      {x});
}

array all_min(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Min),
      {x});
}

array all_gather(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  auto result_shape = x.shape();
  if (result_shape.size() == 0) {
    result_shape.push_back(group.size());
  } else {
    result_shape[0] *= group.size();
  }
  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<AllGather>(stream, group),
      {x});
}

array send(
    const array& x,
    int dst,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot send to a singleton group");
  }
  auto stream = detail::communication_stream(group, s);

  if (dst < 0 || dst >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid destination=" << dst << " for a group of size "
        << group.size();
    throw std::invalid_argument(msg.str());
  }

  return make_chained(group, stream, {x}, [&](std::vector<array> inputs) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<Send>(stream, group, dst),
        std::move(inputs));
  });
}

array recv(
    Shape shape,
    Dtype dtype,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot recv from a singleton group");
  }
  auto stream = detail::communication_stream(group, s);

  if (src < 0 || src >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid source=" << src << " for a group of size " << group.size();
    throw std::invalid_argument(msg.str());
  }

  return make_chained(
      group, stream, std::vector<array>{}, [&](std::vector<array> inputs) {
        return array(
            std::move(shape),
            std::move(dtype),
            std::make_shared<Recv>(stream, group, src),
            std::move(inputs));
      });
}

array recv_like(
    const array& x,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return recv(x.shape(), x.dtype(), src, group_, s);
}

array sum_scatter(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  if (group.size() == 1) {
    return x;
  }
  if (x.shape()[0] % group.size() != 0) {
    std::ostringstream msg;
    msg << "[sum_scatter] Invalid shape=" << x.shape()
        << " for a group of size " << group.size()
        << ". The first dimension (axis 0) must be divisible by the group size.";
    throw std::invalid_argument(msg.str());
  }

  auto result_shape = x.shape();
  result_shape[0] /= group.size();
  auto stream = detail::communication_stream(group, s);

  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<ReduceScatter>(stream, group, ReduceScatter::Sum),
      {x});
}
} // namespace mlx::core::distributed
