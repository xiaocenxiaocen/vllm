# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import List, Dict, Tuple, Union
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
from hidet.ir.type import data_type

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
    warp_reduce_sum,
)

from hidet.ir.cute import auto_layout, layout_auto

from hidet.utils.py import cdiv
from hidet.utils import initialize
from hidet.lang import grid

from hidet.lang.types import i32, f32, f16, u4
from hidet.ir.type import DataType
from hidet.runtime.compiled_module import CompiledFunction
from hidet.ir.library import tune

import torch
from .weight_utils import wrapper_func, bench, hidet_dtype_to_torch_dtype
from .tuning import LinearCache, linear_cache
from .linear_perf_model import LinearConfig as Config, GemmAnalyticalModel


_predefined_config: List[Config] = []


@initialize()
def register_configs():
    PARALLEL_K_PARTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    a = TensorLayout(((4, 8, 2), (2, 2)), ((16, 1, 128), (8, 64)))
    b = TensorLayout(((4, 8, 2), (2, 2, 2)), ((32, 1, 256), (16, 8, 128)))
    c = TensorLayout(((4, 8, 2), (2, 2)), ((2, 8, 0), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 32), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 3, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 6, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 6, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (3, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 32, 4, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    for parallel_k_parts in PARALLEL_K_PARTS:
        _predefined_config.append(Config(tiled_mma, 64, 1, parallel_k_parts))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_config.append(Config(tiled_mma, 32, 3, 1))

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_config.append(Config(tiled_mma, 32, 3, 1))


INST_K = 16


class Gemm:
    def __init__(self, name: str, a_dtype: str, b_dtype: str, k: int, n: int, sm_ver: int = 80, tuning_cache: LinearCache = None):
        self.name = name
        self.a_dtype = data_type(a_dtype)
        self.b_dtype = data_type(b_dtype)
        self.k = k
        self.n = n
        self.functions: Dict[Config, CompiledFunction] = {}
        self.m_symbol_name = f"m_{self.name}"
        self.sm_ver = sm_ver
        self.tuning_cache = tuning_cache

        # cache
        self.cache: Dict[int, Tuple[Config, CompiledFunction]] = {}
        self._compile()
        # self.last_m_value = None
        # self.c_parallel_k_parts = None
        # self.lock = None

    def __call__(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        from hidet.ffi import runtime_api

        m, k = a.shape
        n, k = b.shape
        adtype = hidet_dtype_to_torch_dtype(self.a_dtype)
        bdtype = hidet_dtype_to_torch_dtype(self.b_dtype)

        runtime_api.set_symbol_value(self.m_symbol_name, m)
        m_clip = min(max(m, 8), 128, 1024)
        m_roundup = min(i for i in [8, 16, 32, 48, 64, 128, 1024] if i >= m_clip)
        if m_roundup in self.cache:
            config, func = self.cache[m_roundup]
        else:
            raise NotImplementedError()
        parallel_k_parts = config.parallel_k_parts
        if bias is None:
            bias = torch.zeros((n, ), dtype=bdtype, device="cuda")
        if True:
            bm, bn, _ = config.thread_block_shape
            c = torch.empty((m, n), dtype=adtype, device="cuda")
            c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=adtype, device="cuda")
            grid_m, grid_n = cdiv(m, bm), cdiv(n, bn)
            lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
            wrapper_func(func, a, b, c, bias, c_parallel_k_parts, lock)
            return c
        else: 
            c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=adtype, device="cuda")
            func(a, b, bias, c_parallel_k_parts)
            if parallel_k_parts == 1:
                return c_parallel_k_parts.reshape(m, n)
            else:
                return c_parallel_k_parts.sum(0) + bias
        #import torch.nn.functional as F 
        ##c2 = F.linear(a, b, bias)
        #c2 = a @ b.T + bias
        #cmax = torch.max(torch.abs(c - c2))
        #if cmax > 1e-3:
        #    print(m, n, self.k)
        #    print(config)
        #    print(cmax)
        #    print(c)
        #    print(c - c2)
        #    print(c2)
        #    print(bias)
        #    import numpy as np
        #    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
        #    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=5e-3)
        #return c

    def _create_fake_tensors(self, M: int):
        adtype = hidet_dtype_to_torch_dtype(self.a_dtype)
        bdtype = hidet_dtype_to_torch_dtype(self.b_dtype)
        bias_dtype = hidet_dtype_to_torch_dtype(self.b_dtype)
        K = self.k
        N = self.n
        lo = -3
        hi = 3
        device = "cuda"
        a = torch.randint(low=lo, high=hi, size=(M, K), dtype=adtype, device=device)
        b = torch.randint(low=lo, high=hi, size=(K, N), dtype=bdtype, device=device)
        bias = torch.randint(low=lo, high=hi, size=(N, ), dtype=bias_dtype, device=device)
        return a, b, bias

    def _compile(self):
        modules = tune.extract_ir_modules(self.modules)

        model = GemmAnalyticalModel()
        k = self.k
        n = self.n
        adtype = hidet_dtype_to_torch_dtype(self.a_dtype)
        for M in [8, 16, 32, 48, 64, 128, 1024]:
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
                time = model.predict(cfg, M, n, k)
                module2time[m] = time

            truncate = 4
            sorted_module = sorted(modules, key=lambda m: module2time[m])
            candidates = sorted_module[:truncate]
            min_time = None
            min_cfg = None
            min_func = None
            # a, b, bias
            a, b, bias = self._create_fake_tensors(M)
            c = torch.empty((M, n), dtype=adtype, device="cuda")
            from hidet.ffi import runtime_api

            runtime_api.set_symbol_value(self.m_symbol_name, M)
            for m in candidates:
                cfg = m._tuning_kwargs["config"]
                func = m.build()

                bm, bn, _ = cfg.thread_block_shape
                parallel_k_parts = cfg.parallel_k_parts
                c_parallel_k_parts = torch.empty((parallel_k_parts, M, n), dtype=adtype, device="cuda")
                grid_m, grid_n = cdiv(M, bm), cdiv(n, bn)
                
                def fn():
                    #if parallel_k_parts == 1:
                    #    func(a, b, bias, c_parallel_k_parts)
                    #    return c_parallel_k_parts.reshape(M, n)
                    #else:
                    #    func(a, b, bias, c_parallel_k_parts)
                    #    return c_parallel_k_parts.sum(0) + bias
                    lock = torch.zeros((grid_m, grid_n), dtype=torch.int32, device="cuda")
                    wrapper_func(func, a, b, c, bias, c_parallel_k_parts, lock)

                time = bench(fn)
                if min_time is None or time < min_time:
                    min_time = time
                    min_cfg = cfg
                    min_func = func
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
            return self._single_buffer(config)
        else:
            return self._multi_buffer(config)

    def _single_buffer(self, config: Config):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        bm, bn, _ = config.thread_block_shape
        threads = config.threads
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, f16)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)

        k_partition = self._k_partition(config)
        tune.check(k == parallel_k_parts * k_partition)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: a_dtype[m, k],
                b: b_dtype[n, k],
                c: a_dtype[m, n],
                bias: b_dtype[n],
                c_parallel_k_parts: a_dtype[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bn), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts, cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.y
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.x
                k_start_pos = k_part * k_partition

                ts_a = make_tensor(a_dtype, TensorLayout((bm, bk), (bk, 1)), "shared")
                ts_b = make_tensor(b_dtype, TensorLayout((bn, bk), (bk, 1)), "shared")

                tr_a = make_tensor(a_dtype, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(b_dtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tg_a = tensor_view(a[pid_m * bm:, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn:, k_start_pos:], TensorLayout((bn, k), (k, 1)), "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                cp_async_wait_all()
                syncthreads()

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    copy(auto_copy((bm, bk)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb, msk_b, evict="evict_first")

                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])

                    for ki in grid(k_tile_max, attrs="u+"):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
                    syncthreads()

                k_part = blockIdx.x
                if k_tile > INST_K:
                    tr_c_f16 = cast(warp_reduce_sum(tr_c), a_dtype)
                else:
                    tr_c_f16 = cast(tr_c, a_dtype)

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c_f16, auto_layout, "register")

                #if parallel_k_parts == 1:
                #    msk_bias = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                #    tr_bias = make_tensor(b_dtype, layout_auto((bm, bn), (0, 1)), "register")
                #    tg_bias = tensor_view(bias[pid_n * bn : (pid_n + 1) * bn], TensorLayout((bm, bn), (0, 1)), "global")
                #    txgx_bias = partition_src(tg_bias, auto_copy())
                #    txrx_bias = partition_dst(tr_bias, auto_copy())
                #    copy(auto_copy((bm, bn)), txgx_bias, txrx_bias, msk_bias)
                #    tr_C = tr_bias + tr_C
                # 
                #tg_c = tensor_view(
                #    c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                #    TensorLayout((bm, bn), (n, 1)),
                #    "global",
                #)

                #txrx_c = partition_src(tr_C, auto_copy())
                #txgx_c = partition_dst(tg_c, auto_copy())
                #copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

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
                    tr_c_k_part = make_tensor(a_dtype, auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    # release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    msk_bias = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                    tr_bias = make_tensor(b_dtype, layout_auto((bm, bn), (0, 1)), "register")
                    tg_bias = tensor_view(bias[pid_n * bn : (pid_n + 1) * bn], TensorLayout((bm, bn), (0, 1)), "global")
                    txgx_bias = partition_src(tg_bias, auto_copy())
                    txrx_bias = partition_dst(tr_bias, auto_copy())
                    copy(auto_copy((bm, bn)), txgx_bias, txrx_bias, msk_bias)
                    tr_C = tr_bias + tr_C

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
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        tiled_mma = config.tiled_mma
        bk = config.block_k
        parallel_k_parts = config.parallel_k_parts
        stages = config.stages
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        bm, bn, bk = config.thread_block_shape
        threads = config.threads
        parallel_k_parts = config.parallel_k_parts
        tiled_mma = config.tiled_mma
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, f16)
        compute_capability = self.sm_ver
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        maximum_dynamic_smem_size = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(dynamic_smem_bytes <= maximum_dynamic_smem_size)

        k_partition = self._k_partition(config)
        tune.check(k_partition >= bk * stages)
        tune.check(k == parallel_k_parts * k_partition)
        tune.check(not (bm == 128 and bn == 128 and bk == 128))

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: a_dtype[m, k],
                b: b_dtype[n, k],
                c: a_dtype[m, n],
                bias: b_dtype[n],
                c_parallel_k_parts: a_dtype[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bm), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = parallel_k_parts, cdiv(m, bm) * cdiv(n, bn)
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.y
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.x
                #if k_part == 0 and threadIdx.x == 0:
                #    lock[pid_m, pid_n] = 0
                k_start_pos = k_part * k_partition

                ts_a = make_tensor(a_dtype, TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
                ts_b = make_tensor(b_dtype, TensorLayout((bn, bk, stages), (bk, 1, bn * bk)), "shared")

                tr_a = make_tensor(a_dtype, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(b_dtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn :, k_start_pos:], TensorLayout((bn, k), (k, 1)), "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                for s in range(stages - 1):
                    copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s], msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, s], txsb[:, :, s], msk_b, evict="evict_first")
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb_p = txSb[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in grid(k_tile_max, attrs="u+"):
                        if ki == k_tile_max - 1:
                            # txSa_p = txSa[:, :, :, smem_pipe_read]
                            # txSb_p = txSb[:, :, :, smem_pipe_read]
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])

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
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

                k_part = blockIdx.x

                if k_tile > INST_K:
                    tr_c_f16 = cast(warp_reduce_sum(tr_c), a_dtype)
                else:
                    tr_c_f16 = cast(tr_c, a_dtype)
 
                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_C = rearrange(tr_c_f16, auto_layout, "register")

                #if parallel_k_parts == 1:
                #    msk_bias = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                #    tr_bias = make_tensor(b_dtype, layout_auto((bm, bn), (0, 1)), "register")
                #    tg_bias = tensor_view(bias[pid_n * bn : (pid_n + 1) * bn], TensorLayout((bm, bn), (0, 1)), "global")
                #    txgx_bias = partition_src(tg_bias, auto_copy())
                #    txrx_bias = partition_dst(tr_bias, auto_copy())
                #    copy(auto_copy((bm, bn)), txgx_bias, txrx_bias, msk_bias)
                #    tr_C = tr_bias + tr_C

                #tg_c = tensor_view(
                #    c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                #    TensorLayout((bm, bn), (n, 1)),
                #    "global",
                #)

                #txrx_c = partition_src(tr_C, auto_copy())
                #txgx_c = partition_dst(tg_c, auto_copy())
                #copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

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
                    tr_c_k_part = make_tensor(a_dtype, auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)
                    # release_seq_semaphore(lc, 0)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn,],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    msk_bias = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                    tr_bias = make_tensor(b_dtype, layout_auto((bm, bn), (0, 1)), "register")
                    tg_bias = tensor_view(bias[pid_n * bn : (pid_n + 1) * bn], TensorLayout((bm, bn), (0, 1)), "global")
                    txgx_bias = partition_src(tg_bias, auto_copy())
                    txrx_bias = partition_dst(tr_bias, auto_copy())
                    copy(auto_copy((bm, bn)), txgx_bias, txrx_bias, msk_bias)
                    tr_C = tr_bias + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        return script_module.ir_module()


linear_tuning_cache = linear_cache("linear")


def unload_linear_tuning_cache():
    linear_tuning_cache.save()


atexit.register(unload_linear_tuning_cache)


def w16a16_linear(name: str, a_dtype: Union[str, DataType], b_dtype: Union[str, DataType], input_feats: int, output_feats: int):
    hidet.option.search_space(2)
    return Gemm(name, a_dtype, b_dtype, input_feats, output_feats, tuning_cache=linear_tuning_cache)