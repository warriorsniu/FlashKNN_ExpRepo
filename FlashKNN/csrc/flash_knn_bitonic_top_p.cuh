#pragma once

// Generated bitonic top-P network shared by the production SMPS kernel and
// PS ablation kernels.  Callers retain the lower half of the distributed
// candidate array and place new candidates in the upper half.
namespace flashknn {

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
        distance[Register] = take_peer ? peer_distance : current_distance;
        index[Register] = take_peer ? peer_index : current_index;
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
            const CoordDType current_distance = distance[Register];
            const CoordDType peer_distance = distance[PeerRegister];
            const int current_index = index[Register];
            const int peer_index = index[PeerRegister];
            distance[Register] = swap ? peer_distance : current_distance;
            distance[PeerRegister] = swap ? current_distance : peer_distance;
            index[Register] = swap ? peer_index : current_index;
            index[PeerRegister] = swap ? current_index : peer_index;
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
        const CoordDType current_distance = distance[Register];
        const CoordDType peer_distance = distance[PeerRegister];
        const int current_index = index[Register];
        const int peer_index = index[PeerRegister];
        const bool swap = current_distance > peer_distance;
        distance[Register] = swap ? peer_distance : current_distance;
        distance[PeerRegister] = swap ? current_distance : peer_distance;
        index[Register] = swap ? peer_index : current_index;
        index[PeerRegister] = swap ? current_index : peer_index;
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
        distance[Register] = take_peer ? peer_distance : current_distance;
        index[Register] = take_peer ? peer_index : current_index;
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
            const CoordDType current_distance = distance[Register];
            const CoordDType peer_distance = distance[PeerRegister];
            const int current_index = index[Register];
            const int peer_index = index[PeerRegister];
            const bool swap = current_distance > peer_distance;
            distance[Register] = swap ? peer_distance : current_distance;
            distance[PeerRegister] = swap ? current_distance : peer_distance;
            index[Register] = swap ? peer_index : current_index;
            index[PeerRegister] = swap ? current_index : peer_index;
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

template <typename CoordDType, int ArrayLengthPerThread, int DepthK>
__device__ __forceinline__ void BitonicTopP(
    CoordDType* best_distance, int* best_index) {
    static_assert(ArrayLengthPerThread >= 2);
    static_assert(
        (ArrayLengthPerThread & (ArrayLengthPerThread - 1)) == 0);
    constexpr int LaneWidth = 1 << DepthK;
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
}

}  // namespace flashknn
