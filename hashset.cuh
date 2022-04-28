#ifndef _HASHSET_CUH_
#define _HASHSET_CUH_

#include <cstdint>

constexpr uint32_t KEY_EMPTY = 0xffffffff;

__device__ uint32_t hash(uint32_t k, uint32_t capacity)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity - 1);
}

__device__ void hashset_insert(uint32_t* keys, uint32_t k, int* count, uint32_t capacity)
{
    uint32_t slot = hash(k, capacity);

    while (true) {
        uint32_t prev = atomicCAS(&keys[slot], KEY_EMPTY, k);
        if (prev == k)
            return;
        
        if (prev == KEY_EMPTY) {
            atomicAdd(count, 1);
            return;
        }

        slot = (slot + 1) & (capacity - 1);
    }
}

static inline uint32_t get_hashset_capacity(uint32_t max_size, double load_factor)
{
    auto threshold = uint32_t(max_size / load_factor);
    uint32_t capacity = 1;
    while (capacity < threshold)
        capacity *= 2;
    return capacity;
}

#endif
