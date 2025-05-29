# Performant Hopper Dense GEMM in Python

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.cute.nvgpu.cpasync as cpasync
import torch
from typing import Tuple, Optional

class HopperGemm:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int],
        mma_shape: Tuple[int, int, int],
    ):
        self.cta_tiler = cta_tiler
        self.mma_shape = mma_shape
        # 3 pipelining stages for Hopper GEMMs
        # TODO: replace this with a heuristic
        self.num_ab_pipeline_stages = 3

    def _get_shared_memory_struct(
        self,
        A_layout: cute.Layout,
        B_layout: cute.Layout,
        a_dtype,
        b_dtype,
    ):
        @cute.struct
        class SharedMemory:
            # we will use this for our barrier
            ab_barrier_full: cute.struct.MemRange[cutlass.Int64, self.num_ab_pipeline_stages]
            ab_barrier_empty: cute.struct.MemRange[cutlass.Int64, self.num_ab_pipeline_stages]
            sA: cute.struct.Align[
                cute.struct.MemRange[a_dtype, cute.size(A_layout)],
                1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_dtype, cute.size(B_layout)], 
                1024
            ]
        return SharedMemory

    def _make_shared_memory_layouts(
        self, 
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
    ) -> Tuple[cute.Layout, cute.Layout]:
        # These will be used to load A and B into shared memory.
        A_smemload_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            A.element_type,
        )
        B_smemload_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            B.element_type,
        )

        # Given these atoms, we will now fully specify the shared memory layout.
        # Q: How do we choose how to tile the shared memory layouts?
        # A: You'll need to understand what memory pattern the wgmma expects.
        sA_shape = (self.cta_tiler[0], self.cta_tiler[2], self.num_ab_pipeline_stages)
        sA_layout = cute.tile_to_shape(
            A_smemload_atom,
            sA_shape,
            order=(0,1,2)
        )
        sB_shape = (self.cta_tiler[1], self.cta_tiler[2], self.num_ab_pipeline_stages)
        sB_layout = cute.tile_to_shape(
            B_smemload_atom,
            sB_shape,
            order=(0,1,2)
        )

        return sA_layout, sB_layout

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
    ):
        # /////////////////////////////////////////////
        # 1. Define the shared memory swizzling layouts
        # Note: _get_shared_memory_struct will also allocate
        # the synchronization/barrier memory.
        sA_layout, sB_layout = self._make_shared_memory_layouts(A, B, C)
        shared_memory_struct = self._get_shared_memory_struct(
            sA_layout,
            sB_layout,
            A.element_type,
            B.element_type,
        )
        # /////////////////////////////////////////////

        # /////////////////////////////////////////////
        # 2. Configure our TiledMMAs
        # Very simple! Just tell the wGMMA what the data types are,
        # and how you're planning to feed them into it (specify row-major or column-major)
        # for A and B, and specify whether A is in RMEM or SMEM.
        mma_op = cute.nvgpu.warpgroup.MmaF16BF16Op(
            A.element_type, # AB_dtype
            C.element_type, # Acc_dtype
            self.mma_shape,
            cute.nvgpu.warpgroup.OperandSource.SMEM, 
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        lambda_copy_size = lambda tensor, smem_layout: cute.size_in_bytes(
            tensor.element_type, cute.dice(smem_layout, (1,1,None))
        )
        a_copy_size, b_copy_size = lambda_copy_size(A, sA_layout), lambda_copy_size(B, sB_layout)
        self.num_tma_load_bytes = a_copy_size + b_copy_size

        # /////////////////////////////////////////////

        # /////////////////////////////////////////////
        # 3. Configure our Copies
        # We provide the whole GMEM, and a unit of sMEM that we want to copy it into.
        # we also provide the tiler so that we 
        # cute.dice ensures that we only give it the layout for ONE pipelined stage, as that
        # is our atomic unit of copying.
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tma_tile_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            A, # src tensor
            cute.dice(sA_layout, (1,1,None)), # "dst tile shape"
            (self.cta_tiler[0], self.cta_tiler[2]), # "src tile shape"
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tma_tile_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            B,
            cute.dice(sB_layout, (1,1,None)),
            (self.cta_tiler[1], self.cta_tiler[2]),
        )
        # for the final cop
        # /////////////////////////////////


        
        kernel = self.kernel(
            tiled_mma,
            tma_tensor_a,
            tma_tensor_b,
            C,
            tma_atom_a,
            tma_atom_b,
            sA_layout,
            sB_layout,
            shared_memory_struct,
        ).launch(
            grid=(1,1,1),
            block=(128,1,1),
            smem=shared_memory_struct.size_in_bytes(),
        )
        pass

    @cute.kernel
    def kernel(
        self,
        # Reminder: these are not the raw tensors, but rather the TMA tensors.
        # What are TMA tensors? Try printing out mA and you'll get:
        # <tensor<(0, 0) o (M, K): (1@1, 1@0)>
        # 1@0 are unit vectors. 1@0 is equivalent to (1, 0, ...) and 1@1 is equivalent to (0, 1, ...)
        # so the layout for mA is essentially a mapping from (x, y) -> (y, x)
        tiled_mma: cute.TiledMma,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        shared_memory_struct: cutlass.Constexpr[cute.struct],
    ):
        # /////////////////////////////
        # 1. Allocate the shared memory, then
        # get the specific tensors that we need.
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_memory_struct)
        # note! remember that the smem allocator deals with
        # allocation. we allocate raw bytes. in order to get
        # the tensors, we still need to get the layouts. We may be more 
        # familiar with:
        # sA = cute.make_tensor(
        #    storage.sA.data_ptr(),
        #    sA_layout
        # )
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        # /////////////////////////////

        # /////////////////////////////
        # 2. Prefetch the TMA descriptors
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        # /////////////////////////////

        # /////////////////////////////
        # 3. Set up synchronization.
        producer_group = cutlass.utils.CooperativeGroup(cutlass.utils.Agent.Thread)
        consumer_group = cutlass.utils.CooperativeGroup(cutlass.utils.Agent.Thread)
        ab_pipeline = cutlass.utils.PipelineTmaAsync.create(
            storage.ab_barrier_full.data_ptr(),
            self.num_ab_pipeline_stages,
            producer_group,
            consumer_group,
            self.num_tma_load_bytes,
            cute.make_layout((1,1,1))
        )
        ab_producer_state = cutlass.utils.make_pipeline_state(
            cutlass.utils.PipelineUserType.Producer, self.num_ab_pipeline_stages
        )
        ab_consumer_state = cutlass.utils.make_pipeline_state(
            cutlass.utils.PipelineUserType.Consumer, self.num_ab_pipeline_stages
        )
        # /////////////////////////////


        # /////////////////////////////
        # 4. Get the tiles that we are operating with
        mblk, nblk, kblk = self.cta_tiler
        gA = cute.local_tile(mA, (mblk, kblk), (bidx, None))
        gB = cute.local_tile(mB, (kblk, nblk), (bidy, None))
        gC = cute.local_tile(mC, (mblk, nblk), (bidx, bidy))
        # /////////////////////////////

        # /////////////////////////////
        # 5. "Partition" the tma tiles across the threads.
        # This will only really partition the gC tiles, because
        thr_mma = tiled_mma.get_slice(tidx)
        # in hopper, there is no thread-level splitting for TMAs.
        # for that reason, tAgA and tBgB is the same across the threads
        # concretely we get:
        # Mode 0: (64, 16) [our MMA tile sizes]
        # Mode 1: cta_tile[M_size] // mma_tile[M_size]
        # Mode 2: cta_tile[K_size] // mma_tile[K_size]
        # Mode 3: num k blocks

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cute.Int32(0),
            cute.make_layout((1,)),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2)
        )
        tAsB, tAgB = cpasync.tma_partition(
            tma_atom_b,
            cute.Int32(0),
            cute.make_layout((1,)),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2)
        )

        # /////////////////////////////

        # /////////////////////////////
        # 6. Partition the shared memory across the threads.
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)

        # /////////////////////////////
        # 7. Define "fragments"
        # if you print these out, you will see
        # tensor<Value(%488 = "cute.get_iter"(%486) : (!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,2,4,(1,3)):(0,512,2,(0,1024))">) -> !cute_nvgpu.smem_desc) o (1,2,4,(1,3)):(0,512,2,(0,1024))>
        # this is more an abstraction over an smem descriptor when we're dealing with 
        # TMAs!
        tCrA = thr_mma.make_fragment_A(tCsA)
        tCrB = thr_mma.make_fragment_B(tCsB)
        # Allocate the registers for C
        # TCGC is (2,2,4)
        tCrC = thr_mma.make_fragment_C(tCgC.shape)
        tCrC.fill(0.0)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)


        # ///////////////////////////
        # 8. Prefill: Kick off all TMA loads except the last one in the pipeline.
        for _1 in cutlass.range_dynamic(0, self.num_ab_pipeline_stages- 1, 1, unroll=1):
            if warp_idx == 0:
                ab_pipeline.producer_acquire(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[None, ab_producer_state.count],
                    tAsA[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tAgB[None, ab_producer_state.count],
                    tAsB[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                ab_producer_state.advance()

        # ///////////////////////////
        # 9. Main loop
        NUM_K_BLOCKS = cute.size(tAgA, [1])
        for _2 in cutlass.range_dynamic(0, NUM_K_BLOCKS, 1, unroll=1):
            if warp_idx == 0:
                ab_pipeline.producer_acquire(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[None, ab_producer_state.count],
                    tAsA[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tAgB[None, ab_producer_state.count],
                    tAsB[None, ab_producer_state.index],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                ab_producer_state.advance()

            ab_pipeline.consumer_wait(ab_consumer_state)
            num_kphases = cute.size(tCrA, mode=[2])
            for kphase_idx in range(num_kphases):
                kphase_coord = (None, None, kphase_idx, ab_consumer_state.index)

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[kphase_coord],
                    tCrB[kphase_coord],
                    tCrC,
                )

            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()

        # //////////////////////////////////////









def run_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cta_tiler: Tuple[int, int, int],
    mma_shape: Tuple[int, int, int],
):
    A_dlpack = from_dlpack(A, assumed_align=16)
    B_dlpack = from_dlpack(B, assumed_align=16)
    C_dlpack = from_dlpack(C, assumed_align=16)

    gemm_op = HopperGemm(cta_tiler, mma_shape)
    compiled_kernel = cute.compile(gemm_op, A_dlpack, B_dlpack, C_dlpack)
    compiled_kernel(A_dlpack, B_dlpack, C_dlpack)


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Run GEMM on input matrices.")
    parser.add_argument('--mnkl', default="4096,4096,4096,1", type=str, help='Shape of the matrices to generate')
    parser.add_argument('--cta_tiler', default="128,128,64", type=str, help='CTA tiling size')
    parser.add_argument('--mma_shape', default="64,64,16", type=str, help='WGMMA shape')

    args = parser.parse_args()
    M, N, K, L = map(int, args.mnkl.split(","))
    cta_tiler = tuple(map(int, args.cta_tiler.split(",")))
    mma_shape = tuple(map(int, args.mma_shape.split(",")))

    # Create column-major (Fortran-contiguous) tensors
    A = torch.randn(K, M, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32).permute(1, 0)

    run_gemm(A, B, C, cta_tiler, mma_shape)