#pragma once

namespace MC {

// ����kernel: prepareMatrix
#ifndef P_NTHREADS_X // node corner
#  define P_NTHREADS_X 64
#endif // !P_NTHREADS_X
#ifndef P_NTHREADS_Y // voxel corner
#  define P_NTHREADS_Y 16
#endif // !P_NTHREADS_Y

// ����kernel: determine��tomesh
#ifndef V_NTHREADS
#  define V_NTHREADS 256
#endif // !NTHREADS

#ifndef MAX_NUM_STREAMS
#  define MAX_NUM_STREAMS 512 // ���ڴ����з���--voxel�����ֿ���
#endif // !MAX_NUM_STREAMS

} // namespace MC