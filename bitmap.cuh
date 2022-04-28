#ifndef _BITMAP_CUH_
#define _BITMAP_CUH_

#include <cstdint>

struct Bitmap {
    char *bits;
    int count;

    void insert(uint32_t k)
    {
        auto i = k / 8;
        auto v = 1 << (k & 7);
        auto old = __sync_fetch_and_or(&bits[i], v);
        if (!(old & v))
            __sync_fetch_and_add(&count, 1); 
    }
};

__device__ void bitmap_insert(uint32_t* bits, uint32_t k, int* count)
{
    auto i = k / 32;
    auto v = 1 << (k & 31);
    auto old = atomicOr(&bits[i], v);
    if (!(old & v))
        atomicAdd(count, 1);
}

static inline uint32_t get_bitmap_size(int n)
{
    return (((n + 7) / 8) + 15) & 0xfffffff0;
}

#endif
