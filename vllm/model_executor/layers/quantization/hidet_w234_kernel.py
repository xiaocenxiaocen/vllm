# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import List, Dict, Tuple
import atexit

import hidet

from hidet.lang import attrs
from hidet.lang.cuda import (
    blockIdx,
    threadIdx,
    syncthreads,
    cp_async_wait_all,
    cp_async_commit_group,
    cp_async_wait_group,
)
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.expr import symbol_var

from hidet.ir.cute.layout import TensorLayout, make_layout
from hidet.ir.cute.layout import Level
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    rearrange,
    cast,
    fill,
)

from hidet.ir.cute import auto_layout, layout_auto

from hidet.utils.py import cdiv
from hidet.utils import initialize
from hidet.lang import grid

from hidet.lang.types import i32, f32, f16, u4, u2, u1
from hidet.ir.type import DataType
from hidet.runtime.compiled_module import CompiledFunction
from hidet.ir.library import tune

import torch
from .perf_model import Config, FpAIntBGemmAnalyticalModel
from .weight_utils import bench, wrapper_func
from .tuning import LinearCache, linear_cache


_predefined_config: List[Config] = []


@initialize()
def register_configs():
    PARALLEL_K_PARTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 128, 4, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 256, 1, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 2), TensorLayout((1, 2)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 256, 1, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 128, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 256, 1, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 128, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 128, 2, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (6, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (2, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (8, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (2, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))


class FpAIntSubByteBGemm:
    def __init__(self, name: str, k: int, n: int, group_size: int, weight_bits: int = 3, sm_ver: int = 80, tuning_cache: LinearCache = None):
        self.name = name
        self.k = k
        self.n = n
        self.group_size = group_size
        self.weight_bits = weight_bits
        self.functions: Dict[Config, CompiledFunction] = {}
        self.m_symbol_name = f"m_{self.name}"
        self.sm_ver = sm_ver
        self.tuning_cache = tuning_cache

        # cache
        self.cache: Dict[int, Tuple[Config, CompiledFunction]] = {}
        self._compile()

    def call3(self, a: torch.Tensor, b2: torch.Tensor, b1: torch.Tensor, scale: torch.Tensor):
        from hidet.ffi import runtime_api
        m, _ = a.shape
        _, n1 = b1.shape
        _, n2 = b2.shape
        n1 = n1 * b1.element_size() * 8 // 1
        n2 = n2 * b2.element_size() * 8 // 2
        assert n1 == n2
        n = n1
        
        runtime_api.set_symbol_value(self.m_symbol_name, m)
        m_clip = min(max(m, 8), 1024)
        m_roundup = min(i for i in [8, 1024] if i >= m_clip)
        assert m_roundup in self.cache
        config, func = self.cache[m_roundup]
        bm, bn, _ = config.thread_block_shape
        parallel_k_parts = config.parallel_k_parts
        grid_m, grid_n = cdiv(m, bm), cdiv(n, bn)
        c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
        c = torch.empty((m, n), dtype=torch.float16, device="cuda")
        lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
        wrapper_func(func, a, b2, b1, c, scale, c_parallel_k_parts, lock)
        return c
       
    def call24(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor):
        from hidet.ffi import runtime_api        
        m, _ = a.shape
        _, n = b.shape
        n = n * b.element_size() * 8 // self.weight_bits

        runtime_api.set_symbol_value(self.m_symbol_name, m)
        m_clip = min(max(m, 8), 1024)
        m_roundup = min(i for i in [8, 1024] if i >= m_clip)
        assert m_roundup in self.cache
        config, func = self.cache[m_roundup]
        bm, bn, _ = config.thread_block_shape
        parallel_k_parts = config.parallel_k_parts
        grid_m, grid_n = cdiv(m, bm), cdiv(n, bn)
        c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
        c = torch.empty((m, n), dtype=torch.float16, device="cuda")
        lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
        wrapper_func(func, a, b, c, scale, c_parallel_k_parts, lock)
        return c

    def __call__(self, *args: torch.Tensor):
        if self.weight_bits == 3:
            return self.call3(*args)
        elif self.weight_bits == 4 or self.weight_bits == 2:
            return self.call24(*args)

    def _deduce_gmem_layout(self, block_k: int, block_n: int, stages: int = 1):
        # magic gmem and smem layout that coalesce global memory load and
        # resolve shared memory bank conflict
        #basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((32, 2), (1, 8, 4)))
        basic_block = TensorLayout(((8, 2, 2), (2, 4, 2)), ((64, 2, 8), (1, 16, 4)))
        m_mode, n_mode = basic_block
        n_shape = n_mode.shape + (block_k // n_mode.size(),)
        n_stride = n_mode.stride + (basic_block.cosize(),)
        n_mode_ = TensorLayout(n_shape, n_stride)
        m_shape = m_mode.shape + (block_n // m_mode.size(),)
        cosize = block_k // 32 * basic_block.cosize()
        m_stride = m_mode.stride + (cosize,)
        m_mode_ = TensorLayout(m_shape, m_stride)
        if stages > 1:
            smem_layout = make_layout(m_mode_, n_mode_)
            stage_layout = TensorLayout(stages, smem_layout.cosize())
            smem_layout = make_layout(m_mode_, n_mode_, stage_layout)
        else:
            smem_layout = make_layout(m_mode_, n_mode_)

        n_shape = n_mode.shape + (self.k // n_mode.size(),)
        n_stride = n_mode.stride + (basic_block.cosize(),)
        n_mode_ = TensorLayout(n_shape, n_stride)
        m_shape = m_mode.shape + (block_n // m_mode.size(),)
        cosize = self.k // 32 * basic_block.cosize()
        m_stride = m_mode.stride + (cosize,)
        m_mode_ = TensorLayout(m_shape, m_stride)
        gmem_layout = make_layout(m_mode_, n_mode_)
        return gmem_layout, smem_layout

    def _create_fake_tensors(self, M: int):
        adtype = torch.float16
        bdtype = torch.int32
        scale_dtype = torch.float16
        bias_dtype = torch.float16
        K = self.k
        N = self.n
        group_size = self.group_size
        lo = -3
        hi = 3
        device = "cuda"
        if self.weight_bits == 3:
            factor2 = bdtype.itemsize * 8 // 2
            factor1 = bdtype.itemsize * 8 // 1
            a = torch.randint(low=lo, high=hi, size=(M, K), dtype=adtype, device=device)
            b2 = torch.randint(low=0, high=hi, size=(K, N // factor2), dtype=bdtype, device=device)
            b1 = torch.randint(low=0, high=hi, size=(K, N // factor1), dtype=bdtype, device=device)
            scale = torch.randint(low=-1, high=2, size=(K // group_size, N), dtype=scale_dtype, device=device)
            zeros = torch.randint(low=0, high=hi, size=(K // group_size, N), dtype=bias_dtype, device=device)
            return a, b2, b1, scale, zeros
        else:
            factor = bdtype.itemsize * 8 // self.weight_bits
            a = torch.randint(low=lo, high=hi, size=(M, K), dtype=adtype, device=device)
            b = torch.randint(low=0, high=hi, size=(K, N // factor), dtype=bdtype, device=device)
            scale = torch.randint(low=-1, high=2, size=(K // group_size, N), dtype=scale_dtype, device=device)
            zeros = torch.randint(low=0, high=hi, size=(K // group_size, N), dtype=bias_dtype, device=device)
            return a, b, scale, zeros

    def _compile(self):
        modules = tune.extract_ir_modules(self.modules)

        model = FpAIntBGemmAnalyticalModel()
        k = self.k
        n = self.n
        group_size = self.group_size
        for M in [8, 1024]:
            if self.tuning_cache is not None and self.tuning_cache.contains(M, n, k):
                min_cfg = self.tuning_cache.get(M, n, k)
                for m in modules:
                    if str(m._tuning_kwargs["config"]) == str(min_cfg):
                        self.cache[M] = (min_cfg, m.build())
                        break
                continue

            module2time = {}
            for m in modules:
                cfg = m._tuning_kwargs["config"]
                bm, _, _ = cfg.thread_block_shape
                if M <= 128 and (bm >= 2 * M or bm < M // 2):
                    module2time[m] = 1e9
                    continue
                time = model.predict(cfg, M, n, k, group_size, self.weight_bits)
                module2time[m] = time

            truncate = 10
            sorted_module = sorted(modules, key=lambda m: module2time[m])
            candidates = sorted_module[:truncate]
            min_time = None
            min_cfg = None
            min_func = None
            # a, b, scale, zeros
            if self.weight_bits == 3:
                a, b, b1, scale, zeros = self._create_fake_tensors(M)
            else:
                a, b, scale, zeros = self._create_fake_tensors(M)
            c = torch.empty((M, n), dtype=torch.float16, device="cuda")
            from hidet.ffi import runtime_api

            runtime_api.set_symbol_value(self.m_symbol_name, M)
            for m in candidates:
                cfg = m._tuning_kwargs["config"]
                func = m.build()

                bm, bn, _ = cfg.thread_block_shape
                parallel_k_parts = cfg.parallel_k_parts
                c_parallel_k_parts = torch.empty((parallel_k_parts, M, n), dtype=torch.float16, device="cuda")
                grid_m, grid_n = cdiv(M, bm), cdiv(n, bn)

                if self.weight_bits == 3:
                    def fn():
                        lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
                        wrapper_func(func, a, b, b1, c, scale, c_parallel_k_parts, lock)
                    time = bench(fn)
                else:
                    def fn():
                        lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
                        wrapper_func(func, a, b, c, scale, c_parallel_k_parts, lock)
                    time = bench(fn)
                
                if min_time is None or time < min_time:
                    min_time = time
                    min_cfg = cfg
                    min_func = func
            print("func2", M, n, k, min_cfg)
            self.cache[M] = (min_cfg, min_func)
            if self.tuning_cache is not None:
                self.tuning_cache.put(M, n, k, min_cfg)

    def _k_partition(self, config: Config):
        _, _, bk = config.thread_block_shape
        parallel_k_parts = config.parallel_k_parts
        k = self.k
        if parallel_k_parts == 1:
            return k

        k_partition = bk
        while k_partition * parallel_k_parts < k:
            k_partition += bk
        return k_partition

    @tune.space(1, config=[_predefined_config[0]])
    @tune.space(2, config=_predefined_config)
    def modules(self, config: Config):
        stages = config.stages
        if stages == 1:
            if self.weight_bits == 3:
                return self._single3_buffer(config)
            else:
                return self._single_buffer(config)
        else:
            if self.weight_bits == 3:
                return self._multi3_buffer(config)
            else:
                return self._multi_buffer(config)

    def _single3_buffer(self, config: Config):
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        stages = config.stages
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        bm, bn, _ = config.thread_block_shape
        threads = config.threads
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, u2, group_size, stages) + config.dynamic_smem_bytes(f16, u1, group_size, stages)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout((bn, (group_size, bk // group_size)), (1, (0, bn)))
        else:
            scale_smem_layout = TensorLayout((bn, bk), (1, 0))

        k_partition = self._k_partition(config)
        tune.check(k_partition % group_size == 0)
        tune.check(k == parallel_k_parts * k_partition)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b2: u2[n, k],
                b1: u1[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bn), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts * cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                k_part = blockIdx.x % parallel_k_parts
                pid = blockIdx.x // parallel_k_parts
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_start_pos = k_part * k_partition
                k_start_ofs2 = gmem_layout((0, k_start_pos))
                k_start_ofs1 = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk), (bk, 1)), "shared")
                ts_b2 = make_tensor(u2, smem_layout, "shared")
                ts_b1 = make_tensor(u1, smem_layout, "shared")
                ts_scale = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b2 = make_tensor(u2, layout_auto((bn, k_tile * 2)), "register")
                tr_b1 = make_tensor(u1, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float16", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")
                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b2 = tensor_view(b2[pid_n * bn :, k_start_ofs2:], gmem_layout, "global")
                tg_b1 = tensor_view(b1[pid_n * bn :, k_start_ofs1:], gmem_layout, "global")
                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb2 = partition_src(tg_b2, auto_copy())
                txsb2 = partition_dst(ts_b2, auto_copy())
                txgb1 = partition_src(tg_b1, auto_copy())
                txsb1 = partition_dst(ts_b1, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb2 = partition_src(ts_b2, auto_copy())
                txrb2 = partition_dst(tr_b2, auto_copy())
                txSb1 = partition_src(ts_b1, auto_copy())
                txrb1 = partition_dst(tr_b1, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b2 = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_b1 = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                cp_async_wait_all()
                syncthreads()

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    copy(auto_copy((bm, bk)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((bn, bk)), txgb2[:, :, ko], txsb2, msk_b2, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgb1[:, :, ko], txsb1, msk_b1, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgsc[:, :, ko], txssc, msk_scale)

                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb2[:, :, 0], txrb2[:, :, 0])
                    copy(auto_copy(), txSb1[:, :, 0], txrb1[:, :, 0])

                    copy(auto_copy(), txSsc[:, :, 0], txrsc[:, :, 0])

                    for ki in grid(k_tile_max, attrs="u+"):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb2[:, :, ki + 1], txrb2[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb1[:, :, ki + 1], txrb1[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSsc[:, :, ki + 1], txrsc[:, :, (ki + 1) % 2])

                        txrb2_f16 = cast(txrb2[:, :, ki % 2], f16)
                        txrb1_f16 = cast(txrb1[:, :, ki % 2], f16)
                        txrb_f16 = txrsc[:, :, ki % 2] * (f16(2.0) * txrb2_f16 + txrb1_f16)
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)
                    syncthreads()

                k_part = blockIdx.x % parallel_k_parts

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:
                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        return script_module.ir_module()


    def _single_buffer(self, config: Config):
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        stages = config.stages
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        if self.weight_bits == 2:
            wdtype = u2
        else:
            assert self.weight_bits == 4
            wdtype = u4
        bm, bn, _ = config.thread_block_shape
        threads = config.threads
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, wdtype, group_size, stages)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout((bn, (group_size, bk // group_size)), (1, (0, bn)))
        else:
            scale_smem_layout = TensorLayout((bn, bk), (1, 0))

        k_partition = self._k_partition(config)
        tune.check(k_partition % group_size == 0)
        tune.check(k == parallel_k_parts * k_partition)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b: wdtype[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bn), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts * cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                k_part = blockIdx.x % parallel_k_parts
                pid = blockIdx.x // parallel_k_parts
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk), (bk, 1)), "shared")
                ts_b = make_tensor(wdtype, smem_layout, "shared")

                ts_scale = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(wdtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float16", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn :, k_start_ofs:], gmem_layout, "global")

                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                cp_async_wait_all()
                syncthreads()

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    copy(auto_copy((bm, bk)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb, msk_b, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgsc[:, :, ko], txssc, msk_scale)

                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])
                    copy(auto_copy(), txSsc[:, :, 0], txrsc[:, :, 0])

                    for ki in range(k_tile_max):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                            copy(auto_copy(), txSsc[:, :, ki + 1], txrsc[:, :, (ki + 1) % 2])

                        txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], f16))
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)
                    syncthreads()

                k_part = blockIdx.x % parallel_k_parts

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:
                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        return script_module.ir_module()

    def _multi3_buffer(self, config: Config):
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        stages = config.stages
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        bm, bn, bk = config.thread_block_shape
        threads = config.threads
        parallel_k_parts = config.parallel_k_parts
        tiled_mma = config.tiled_mma
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, u2, group_size, stages) + config.dynamic_smem_bytes(f16, u1, group_size, stages)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn, stages)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages), (1, (0, bn), bn * bk // group_size)
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        k_partition = self._k_partition(config)
        tune.check(k_partition >= bk * stages)
        tune.check(k_partition % group_size == 0)
        tune.check(k == parallel_k_parts * k_partition)
        tune.check(not (bm == 128 and bn == 128 and bk == 128))

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b2: u2[n, k],
                b1: u1[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bm), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts * cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                k_part = blockIdx.x % parallel_k_parts
                pid = blockIdx.x // parallel_k_parts
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout((0, k_start_pos))
                k_start_ofs2 = gmem_layout((0, k_start_pos))
                k_start_ofs1 = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
                ts_b2 = make_tensor(u2, smem_layout, "shared")
                ts_b1 = make_tensor(u1, smem_layout, "shared")
                ts_scale = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b2 = make_tensor(u2, layout_auto((bn, k_tile * 2)), "register")
                tr_b1 = make_tensor(u1, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float16", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b2 = tensor_view(b2[pid_n * bn :, k_start_ofs2:], gmem_layout, "global")
                tg_b1 = tensor_view(b1[pid_n * bn :, k_start_ofs1:], gmem_layout, "global")

                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb2 = partition_src(tg_b2, auto_copy())
                txsb2 = partition_dst(ts_b2, auto_copy())
                txgb1 = partition_src(tg_b1, auto_copy())
                txsb1 = partition_dst(ts_b1, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b2 = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_b1 = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                for s in range(stages - 1):
                    copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s], msk_a)
                    copy(auto_copy((bn, bk)), txgb2[:, :, s], txsb2[:, :, s], msk_b2, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgb1[:, :, s], txsb1[:, :, s], msk_b1, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgsc[:, :, s], txssc[:, :, s], msk_scale)
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb2 = partition_src(ts_b2, auto_copy())
                txrb2 = partition_dst(tr_b2, auto_copy())
                txSb1 = partition_src(ts_b1, auto_copy())
                txrb1 = partition_dst(tr_b1, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb2_p = txSb2[:, :, :, smem_pipe_read]
                txSb1_p = txSb1[:, :, :, smem_pipe_read]
                txSsc_p = txSsc[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb2_p[:, :, 0], txrb2[:, :, 0])
                copy(auto_copy(), txSb1_p[:, :, 0], txrb1[:, :, 0])
                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        if ki == k_tile_max - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb2[:, :, k_tile_next, smem_pipe_read], txrb2[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb1[:, :, k_tile_next, smem_pipe_read], txrb1[:, :, (ki + 1) % 2])
                        # TODO: automate this in compiler pass
                        copy(auto_copy(), txSsc[:, :, k_tile_next, smem_pipe_read], txrsc[:, :, (ki + 1) % 2])

                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write], msk_a
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb2[:, :, ko + stages - 1],
                                    txsb2[:, :, smem_pipe_write],
                                    msk_b2,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb1[:, :, ko + stages - 1],
                                    txsb1[:, :, smem_pipe_write],
                                    msk_b1,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        txrb2_f16 = cast(txrb2[:, :, ki % 2], f16)
                        txrb1_f16 = cast(txrb1[:, :, ki % 2], f16)
                        txrb_f16 = txrsc[:, :, ki % 2] * (f16(2.0) * txrb2_f16 + txrb1_f16) 
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)

                k_part = blockIdx.x % parallel_k_parts

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]

                if k_part < parallel_k_parts - 1:
                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        return script_module.ir_module()

    def _multi_buffer(self, config: Config):
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        stages = config.stages
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        bm, bn, bk = config.thread_block_shape
        threads = config.threads
        parallel_k_parts = config.parallel_k_parts
        tiled_mma = config.tiled_mma
        k_tile = config.k_tile
        if self.weight_bits == 2:
            wdtype = u2
        else:
            assert self.weight_bits == 4
            wdtype = u4
        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, wdtype, group_size, stages)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn, stages)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages), (1, (0, bn), bn * bk // group_size)
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        k_partition = self._k_partition(config)
        tune.check(k_partition >= bk * stages)
        tune.check(k_partition % group_size == 0)
        tune.check(k == parallel_k_parts * k_partition)
        tune.check(not (bm == 128 and bn == 128 and bk == 128))

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b: wdtype[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bm), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts * cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                k_part = blockIdx.x % parallel_k_parts
                pid = blockIdx.x // parallel_k_parts
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
                ts_b = make_tensor(wdtype, smem_layout, "shared")

                ts_scale = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(wdtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float16", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn :, k_start_ofs:], gmem_layout, "global")

                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                for s in range(stages - 1):
                    copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s], msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, s], txsb[:, :, s], msk_b, evict="evict_first")
                    copy(auto_copy((bn, bk)), txgsc[:, :, s], txssc[:, :, s], msk_scale)
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb_p = txSb[:, :, :, smem_pipe_read]

                txSsc_p = txSsc[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in range(k_tile_max):
                        if ki == k_tile_max - 1:
                            # txSa_p = txSa[:, :, :, smem_pipe_read]
                            # txSb_p = txSb[:, :, :, smem_pipe_read]
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        # TODO: automate this in compiler pass
                        copy(auto_copy(), txSsc[:, :, k_tile_next, smem_pipe_read], txrsc[:, :, (ki + 1) % 2])

                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write], msk_a
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgb[:, :, ko + stages - 1],
                                    txsb[:, :, smem_pipe_write],
                                    msk_b,
                                    evict="evict_first",
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        txrb_f16 = txrsc[:, :, ki % 2] * cast(txrb[:, :, ki % 2], f16)
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)

                k_part = blockIdx.x % parallel_k_parts

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]

                if k_part < parallel_k_parts - 1:
                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:
                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        return script_module.ir_module()


gptq2_tuning_cache = linear_cache("gptq2")
gptq3_tuning_cache = linear_cache("gptq3")
gptq4_tuning_cache = linear_cache("gptq4")


def unload_tuning_cache():
    gptq2_tuning_cache.save()
    gptq3_tuning_cache.save()
    gptq4_tuning_cache.save()


atexit.register(unload_tuning_cache)


def w2a16_linear(name: str, input_feats: int, output_feats: int, group_size: int):
    hidet.option.search_space(2)
    return FpAIntSubByteBGemm(name, input_feats, output_feats, group_size, 2, tuning_cache=gptq2_tuning_cache)


def w3a16_linear(name: str, input_feats: int, output_feats: int, group_size: int):
    hidet.option.search_space(2)
    return FpAIntSubByteBGemm(name, input_feats, output_feats, group_size, 3, tuning_cache=gptq3_tuning_cache)


def w4a16_linear(name: str, input_feats: int, output_feats: int, group_size: int):
    hidet.option.search_space(2)
    return FpAIntSubByteBGemm(name, input_feats, output_feats, group_size, 4, tuning_cache=gptq4_tuning_cache)
