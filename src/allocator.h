#pragma once
#include <cinttypes>
#include <memory_resource>
#include <memory>
#include <atomic>

// Linear allocator that can be used with std::pmr containers and
// as stand alone allocator

inline size_t alignUpPowerOfTwo(size_t value, size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

struct LinearAllocator : public std::pmr::memory_resource {
  LinearAllocator(void *buffer, size_t size) {
    _buffer = buffer;
    _size = size;
    _offset = 0;
  }

  void *do_allocate(size_t bytes, size_t alignment) override {
    size_t alignedOffset = 0;
    size_t currentOffset = 0;
    do {
      currentOffset = _offset;
      alignedOffset = alignUpPowerOfTwo(_offset, alignment);
    } while (!_offset.compare_exchange_weak(currentOffset, alignedOffset + bytes));
    return (char *)_buffer + alignedOffset;
  }

  void do_deallocate(void *, std::size_t, std::size_t) override {
  }

  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

  void *_buffer;
  std::atomic<size_t> _offset;
  size_t _size;
};

