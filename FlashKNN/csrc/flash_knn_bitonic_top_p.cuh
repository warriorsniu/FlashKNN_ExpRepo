#pragma once

// Generated bitonic top-P network shared by the production SMPS kernel and
// PS ablation kernels.  Callers retain the lower half of the distributed
// candidate array and place new candidates in the upper half.
namespace flashknn {

// CUDA 11.x lowers the short conditional update to predicated moves after
// register allocation, which keeps the fully unrolled network compact. CUDA
// 12.x may preserve the same source form as divergent control flow on sm_89;
// use an in-place mask there to make the branch-free intent explicit.
__device__ __forceinline__ void SelectPeer(
    float& current_distance, int& current_index,
    const float peer_distance, const int peer_index,
    const bool take_peer) {
#if __CUDACC_VER_MAJOR__ < 12
    if (take_peer) {
        current_distance = peer_distance;
        current_index = peer_index;
    }
#else
    const unsigned int mask =
        0u - static_cast<unsigned int>(take_peer);
    unsigned int current_bits = __float_as_uint(current_distance);
    current_bits ^= (current_bits ^ __float_as_uint(peer_distance)) & mask;
    current_distance = __uint_as_float(current_bits);
    current_index ^= (current_index ^ peer_index) & static_cast<int>(mask);
#endif
}

__device__ __forceinline__ void ConditionalSwap(
    float& first_distance, int& first_index,
    float& second_distance, int& second_index,
    const bool swap) {
    const float old_first_distance = first_distance;
    const int old_first_index = first_index;
    first_distance = swap ? second_distance : old_first_distance;
    first_index = swap ? second_index : old_first_index;
    second_distance = swap ? old_first_distance : second_distance;
    second_index = swap ? old_first_index : second_index;
}

template <typename CoordDType, int Register, int RegisterEnd,
          int RegisterBegin, int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void SortDescendingShuffle(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int LocalRegister = Register - RegisterBegin;
        const int logical_index = LocalRegister * LaneWidth + threadIdx.x;
        CoordDType current_distance = distance[Register];
        int current_index = index[Register];
        CoordDType peer_distance = WARP_SHFL(
            current_distance, threadIdx.x ^ Stride, LaneWidth);
        int peer_index = WARP_SHFL(
            current_index, threadIdx.x ^ Stride, LaneWidth);
        const bool ascending_group = (logical_index & Size) != 0;
        const bool lower_lane = (logical_index & Stride) == 0;
        const bool take_peer = ascending_group
            ? (lower_lane ? current_distance > peer_distance
                          : current_distance < peer_distance)
            : (lower_lane ? current_distance < peer_distance
                          : current_distance > peer_distance);
        SelectPeer(
            distance[Register], index[Register],
            peer_distance, peer_index, take_peer);
        SortDescendingShuffle<
            CoordDType, Register + 1, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int RegisterBegin, int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void SortDescendingRegisters(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int LocalRegister = Register - RegisterBegin;
        constexpr int RegisterStride = Stride / LaneWidth;
        if constexpr ((LocalRegister & RegisterStride) == 0) {
            constexpr int PeerRegister = Register + RegisterStride;
            constexpr bool AscendingGroup =
                ((LocalRegister * LaneWidth) & Size) != 0;
            const bool swap = AscendingGroup
                ? distance[Register] > distance[PeerRegister]
                : distance[Register] < distance[PeerRegister];
            ConditionalSwap(
                distance[Register], index[Register],
                distance[PeerRegister], index[PeerRegister], swap);
        }
        SortDescendingRegisters<
            CoordDType, Register + 1, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
}

template <typename CoordDType, int RegisterBegin, int RegisterEnd,
          int LaneWidth, int Size, int Stride>
__device__ __forceinline__ void SortDescendingStrides(
    CoordDType* distance, int* index) {
    if constexpr (Stride < LaneWidth) {
        SortDescendingShuffle<
            CoordDType, RegisterBegin, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    } else {
        SortDescendingRegisters<
            CoordDType, RegisterBegin, RegisterEnd, RegisterBegin,
            LaneWidth, Size, Stride>(distance, index);
    }
    if constexpr (Stride > 1) {
        SortDescendingStrides<
            CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
            Size, Stride / 2>(distance, index);
    }
}

template <typename CoordDType, int RegisterBegin, int RegisterEnd,
          int LaneWidth, int Size = 2>
__device__ __forceinline__ void SortDescending(
    CoordDType* distance, int* index) {
    constexpr int Length = (RegisterEnd - RegisterBegin) * LaneWidth;
    SortDescendingStrides<
        CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
        Size, Size / 2>(distance, index);
    if constexpr (Size < Length) {
        SortDescending<
            CoordDType, RegisterBegin, RegisterEnd, LaneWidth,
            Size * 2>(distance, index);
    }
}

template <typename CoordDType, int Register, int HalfRegisters>
__device__ __forceinline__ void CompareSplit(
    CoordDType* distance, int* index) {
    if constexpr (Register < HalfRegisters) {
        constexpr int PeerRegister = Register + HalfRegisters;
        const CoordDType peer_distance = distance[PeerRegister];
        const int peer_index = index[PeerRegister];
        const bool take_peer = distance[Register] > peer_distance;
        SelectPeer(
            distance[Register], index[Register],
            peer_distance, peer_index, take_peer);
        CompareSplit<CoordDType, Register + 1, HalfRegisters>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int LaneWidth, int Stride>
__device__ __forceinline__ void MergeAscendingShuffle(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        const int logical_index = Register * LaneWidth + threadIdx.x;
        CoordDType current_distance = distance[Register];
        int current_index = index[Register];
        CoordDType peer_distance = WARP_SHFL(
            current_distance, threadIdx.x ^ Stride, LaneWidth);
        int peer_index = WARP_SHFL(
            current_index, threadIdx.x ^ Stride, LaneWidth);
        const bool lower_lane = (logical_index & Stride) == 0;
        const bool take_peer = lower_lane
            ? current_distance > peer_distance
            : current_distance < peer_distance;
        SelectPeer(
            distance[Register], index[Register],
            peer_distance, peer_index, take_peer);
        MergeAscendingShuffle<
            CoordDType, Register + 1, RegisterEnd,
            LaneWidth, Stride>(distance, index);
    }
}

template <typename CoordDType, int Register, int RegisterEnd,
          int LaneWidth, int Stride>
__device__ __forceinline__ void MergeAscendingRegisters(
    CoordDType* distance, int* index) {
    if constexpr (Register < RegisterEnd) {
        constexpr int RegisterStride = Stride / LaneWidth;
        if constexpr ((Register & RegisterStride) == 0) {
            constexpr int PeerRegister = Register + RegisterStride;
            const bool swap =
                distance[Register] > distance[PeerRegister];
            ConditionalSwap(
                distance[Register], index[Register],
                distance[PeerRegister], index[PeerRegister], swap);
        }
        MergeAscendingRegisters<
            CoordDType, Register + 1, RegisterEnd,
            LaneWidth, Stride>(distance, index);
    }
}

template <typename CoordDType, int RegisterEnd, int LaneWidth, int Stride>
__device__ __forceinline__ void MergeAscending(
    CoordDType* distance, int* index) {
    if constexpr (Stride < LaneWidth) {
        MergeAscendingShuffle<
            CoordDType, 0, RegisterEnd, LaneWidth,
            Stride>(distance, index);
    } else {
        MergeAscendingRegisters<
            CoordDType, 0, RegisterEnd, LaneWidth,
            Stride>(distance, index);
    }
    if constexpr (Stride > 1) {
        MergeAscending<
            CoordDType, RegisterEnd, LaneWidth,
            Stride / 2>(distance, index);
    }
}

template <typename CoordDType, int ArrayLengthPerThread, int LaneWidth>
__device__ __forceinline__ void BitonicTopPGeneric(
    CoordDType* best_distance, int* best_index) {
    constexpr int HalfRegisters = ArrayLengthPerThread / 2;
    constexpr int HalfLength = HalfRegisters * LaneWidth;

    #pragma unroll 1
    for (int size = 2; size <= HalfLength; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride < LaneWidth) {
                #pragma unroll
                for (int local_register = 0;
                     local_register < HalfRegisters; ++local_register) {
                    const int reg = HalfRegisters + local_register;
                    const int logical_index =
                        local_register * LaneWidth + threadIdx.x;
                    const CoordDType current_distance = best_distance[reg];
                    const int current_index = best_index[reg];
                    const CoordDType peer_distance = WARP_SHFL(
                        current_distance, threadIdx.x ^ stride, LaneWidth);
                    const int peer_index = WARP_SHFL(
                        current_index, threadIdx.x ^ stride, LaneWidth);
                    const bool ascending_group =
                        (logical_index & size) != 0;
                    const bool lower_lane =
                        (logical_index & stride) == 0;
                    const bool take_peer = ascending_group
                        ? (lower_lane
                            ? current_distance > peer_distance
                            : current_distance < peer_distance)
                        : (lower_lane
                            ? current_distance < peer_distance
                            : current_distance > peer_distance);
                    SelectPeer(
                        best_distance[reg], best_index[reg],
                        peer_distance, peer_index, take_peer);
                }
            } else {
                const int register_stride = stride / LaneWidth;
                const int register_shift = __ffs(register_stride) - 1;
                #pragma unroll
                for (int pair = 0; pair < HalfRegisters / 2; ++pair) {
                    const int first_local =
                        ((pair >> register_shift) << (register_shift + 1))
                        | (pair & (register_stride - 1));
                    const int second_local = first_local + register_stride;
                    const int first = HalfRegisters + first_local;
                    const int second = HalfRegisters + second_local;
                    const int logical_index =
                        first_local * LaneWidth + threadIdx.x;
                    const bool ascending_group =
                        (logical_index & size) != 0;
                    const bool swap = ascending_group
                        ? best_distance[first] > best_distance[second]
                        : best_distance[first] < best_distance[second];
                    ConditionalSwap(
                        best_distance[first], best_index[first],
                        best_distance[second], best_index[second], swap);
                }
            }
        }
    }

    #pragma unroll
    for (int reg = 0; reg < HalfRegisters; ++reg) {
        SelectPeer(
            best_distance[reg], best_index[reg],
            best_distance[reg + HalfRegisters],
            best_index[reg + HalfRegisters],
            best_distance[reg] > best_distance[reg + HalfRegisters]);
    }

    #pragma unroll 1
    for (int stride = HalfLength >> 1; stride > 0; stride >>= 1) {
        if (stride < LaneWidth) {
            #pragma unroll
            for (int reg = 0; reg < HalfRegisters; ++reg) {
                const CoordDType current_distance = best_distance[reg];
                const int current_index = best_index[reg];
                const CoordDType peer_distance = WARP_SHFL(
                    current_distance, threadIdx.x ^ stride, LaneWidth);
                const int peer_index = WARP_SHFL(
                    current_index, threadIdx.x ^ stride, LaneWidth);
                const bool lower_lane = (threadIdx.x & stride) == 0;
                const bool take_peer = lower_lane
                    ? current_distance > peer_distance
                    : current_distance < peer_distance;
                SelectPeer(
                    best_distance[reg], best_index[reg],
                    peer_distance, peer_index, take_peer);
            }
        } else {
            const int register_stride = stride / LaneWidth;
            const int register_shift = __ffs(register_stride) - 1;
            #pragma unroll
            for (int pair = 0; pair < HalfRegisters / 2; ++pair) {
                const int first =
                    ((pair >> register_shift) << (register_shift + 1))
                    | (pair & (register_stride - 1));
                const int second = first + register_stride;
                ConditionalSwap(
                    best_distance[first], best_index[first],
                    best_distance[second], best_index[second],
                    best_distance[first] > best_distance[second]);
            }
        }
    }
}

template <typename CoordDType, int ArrayLengthPerThread, int DepthK>
__device__ __forceinline__ void BitonicTopP(
    CoordDType* best_distance, int* best_index) {
    static_assert(ArrayLengthPerThread >= 2);
    static_assert(
        (ArrayLengthPerThread & (ArrayLengthPerThread - 1)) == 0);
    constexpr int LaneWidth = 1 << DepthK;
    if constexpr (ArrayLengthPerThread == 2) {
        constexpr int HalfRegisters = ArrayLengthPerThread / 2;
        constexpr int RetainedLength = HalfRegisters * LaneWidth;
        SortDescending<
            CoordDType, HalfRegisters, ArrayLengthPerThread,
            LaneWidth>(best_distance, best_index);
        CompareSplit<
            CoordDType, 0, HalfRegisters>(best_distance, best_index);
        MergeAscending<
            CoordDType, HalfRegisters, LaneWidth,
            RetainedLength / 2>(best_distance, best_index);
    } else {
        BitonicTopPGeneric<
            CoordDType, ArrayLengthPerThread, LaneWidth>(
                best_distance, best_index);
    }
}

}  // namespace flashknn
