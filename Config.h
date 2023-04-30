#pragma once

#if defined(__clang__) || defined(__GNUC__)

#define _FORCE_INLINE_ __attribute__((always_inline))

#elif defined(_MSC_VER)

#define _FORCE_INLINE_ __forceinline

#endif