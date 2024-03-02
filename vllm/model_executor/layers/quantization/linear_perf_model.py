# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import torch
import hidet
from hidet.ir.dtypes import DataType, f16, u4

from hidet.ir.cute.layout import TensorLayout
from hidet.ir.cute.algorithm import TiledMma
from hidet.ir.cute import composition

from hidet.utils.py import cdiv

from .weight_utils import canonicalize
from .perf_model import AnalyticalModel, get_num_subcores, get_max_simd_tflops, get_max_tensorcore_tflops, get_max_dram_gbps, get_max_smem_gbps, get_tensorcore_tflops, get_simd_tflops, get_dram_gbps_per_cta, get_smem_gbps, nvsmi


class LinearConfig:
    def __init__(self, tiled_mma: TiledMma, block_k: int, stages: int, parallel_k_slices: int = 1):
        self.tiled_mma = tiled_mma
        self.block_k = block_k
        a_shape, _ = self.tiled_mma.a_tv_layout()
        b_shape, _ = self.tiled_mma.b_tv_layout()

        block_m, k_tile = a_shape
        block_n, _ = b_shape
        self.block_m = block_m
        self.block_n = block_n
        self.parallel_k_slices = parallel_k_slices
        self.k_tile = k_tile
        self._stages = stages

    def __str__(self):
        indent = " " * 2
        return (
            "{\n"
            + f"{indent}tiled_mma: {self.tiled_mma.str_indented(2)},\n"
            + f"{indent}block_k: {self.block_k},\n"
            + f"{indent}stages: {self.stages},\n"
            + f"{indent}parallel_k_parts: {self.parallel_k_parts},\n"
            + "}"
        )

    @property
    def threads(self):
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        c_t, _ = canonicalize(c_tv_layout)
        return c_t.size()

    @property
    def a_elements(self):
        _, a_tv_layout = self.tiled_mma.a_tv_layout()
        _, a_v = canonicalize(a_tv_layout)
        return a_v.size()

    @property
    def b_elements(self):
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, b_v = canonicalize(b_tv_layout)
        return b_v.size()

    @property
    def c_elements(self):
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        _, c_v = canonicalize(c_tv_layout)
        return c_v.size()

    @property
    def parallel_k_parts(self):
        return self.parallel_k_slices

    @property
    def thread_block_shape(self):
        return self.block_m, self.block_n, self.block_k

    @property
    def stages(self):
        return self._stages

    def dynamic_smem_bytes_per_stage(self, a_dtype: DataType, b_dtype: DataType):
        smem_a = self.block_m * self.block_k * a_dtype.nbytes
        smem_b = self.block_n * self.block_k * b_dtype.nbytes
        return smem_a + smem_b

    def dynamic_smem_bytes(self, a_dtype: DataType, b_dtype: DataType):
        dyn_smem_bytes_per_stage = self.dynamic_smem_bytes_per_stage(a_dtype, b_dtype)
        return dyn_smem_bytes_per_stage * self.stages


class GemmAnalyticalModel(AnalyticalModel):
    def __init__(self):
        self.cur_sm_clock = nvsmi(["clocks.max.sm"])[0]
        self.max_tensorcore_tflops = get_max_tensorcore_tflops(torch.float16, self.cur_sm_clock, 0)
        self.max_simd_tflops = get_max_simd_tflops(torch.float32, self.cur_sm_clock, 0)
        self.max_dram_gbps = get_max_dram_gbps(0)
        self.max_smem_gbps = get_max_smem_gbps(0)

    def predict(self, config: LinearConfig, m: int, n: int, k: int, *args) -> float:
        from hidet.ir.dtypes import f16, u4, f32

        adtype = f16
        bdtype = f16
        cdtype = f32

        prop = hidet.cuda.properties()
        num_sm = prop.multiProcessorCount
        block_m, block_n, block_k = config.thread_block_shape
        num_threads = config.threads
        k_tile = config.k_tile
        num_warps = num_threads // 32
        num_cta_m = (m + block_m - 1) // block_m
        num_cta_n = (n + block_n - 1) // block_n
        num_cta_k = config.parallel_k_parts
        num_ctas = num_cta_m * num_cta_n * num_cta_k
        M, N = max(m, block_m), max(n, block_n)

        stages = config.stages
        smem_per_sm = prop.sharedMemPerMultiprocessor
        regs_per_sm = prop.regsPerMultiprocessor
        threads_per_sm = prop.maxThreadsPerMultiProcessor
        smem_per_cta_per_stage = (
            block_m * block_k * adtype.nbytes
            + block_n * block_k * bdtype.nbytes
        )
        smem_per_cta = smem_per_cta_per_stage * stages
        reg_count_per_thread = (
            config.a_elements * 2 * adtype.nbytes
            + config.b_elements * 2 * bdtype.nbytes
            + config.c_elements * cdtype.nbytes
        ) // f32.nbytes
        # over-estimate the registers per thread
        reg_count_per_thread = reg_count_per_thread if reg_count_per_thread >= 128 else 128
        reg_count_per_cta = num_threads * reg_count_per_thread
        num_ctas_per_sm = min(smem_per_sm // smem_per_cta, regs_per_sm // reg_count_per_cta)
        num_ctas_per_sm = min(num_ctas_per_sm, prop.maxBlocksPerMultiProcessor)
        num_ctas_per_sm = min(num_ctas_per_sm, threads_per_sm // num_threads)

        ctas_per_wave = int(num_ctas_per_sm * num_sm)
        num_waves = (num_ctas + ctas_per_wave - 1) // ctas_per_wave

        tc_tput_per_cta = get_tensorcore_tflops(ctas_per_wave, num_warps, 0, self.max_tensorcore_tflops) / ctas_per_wave
        simd_tput_per_cta = get_simd_tflops(ctas_per_wave, num_warps, 0, self.max_simd_tflops) / ctas_per_wave
        active_sm_per_wave = num_sm if num_ctas >= ctas_per_wave else cdiv(num_ctas, num_ctas_per_sm)
        dram_gbps_per_cta = get_dram_gbps_per_cta(active_sm_per_wave, self.max_dram_gbps)
        l2_gbps_per_cta = dram_gbps_per_cta * 4
        smem_gbps_per_cta = get_smem_gbps(1, num_warps, 0, self.max_smem_gbps)
        load_bw = dram_gbps_per_cta

        stages = config.stages
        # a + b + scale + zeros
        gmem_load_per_stage = (
            block_m * block_k * adtype.nbytes
            + block_n * block_k * bdtype.nbytes
        )
        if stages == 1:
            gmem_load_prologue = gmem_load_per_stage

            latency_prologue = 0.0
            compute_simd_per_tile = 2 * block_n * k_tile
            compute_tc_per_tile = 2 * block_m * block_n * k_tile
            latency_simd_per_tile = compute_simd_per_tile / simd_tput_per_cta / 1e12
            latency_tc_per_tile = compute_tc_per_tile / tc_tput_per_cta / 1e12
            # rough estimation
            num_tiles = block_k // k_tile
            latency_compute_per_tile = latency_tc_per_tile + latency_simd_per_tile
            smem_load_per_tile = (
                block_m * k_tile * adtype.nbytes
                + block_n * k_tile * bdtype.nbytes
            )
            latency_smem_per_tile = smem_load_per_tile / smem_gbps_per_cta / 1e9
            if latency_smem_per_tile <= latency_compute_per_tile:
                latency_smem_use = latency_smem_per_tile + latency_compute_per_tile * num_tiles
            else:
                latency_smem_use = latency_smem_per_tile * num_tiles + latency_compute_per_tile

            latency_load_per_stage = (gmem_load_per_stage / load_bw + gmem_load_per_stage / smem_gbps_per_cta) / 1e9
            parallel_k_parts = config.parallel_k_parts
            k_partition = block_k
            while k_partition * parallel_k_parts < k:
                k_partition += block_k
            num_kblocks = (k_partition + block_k - 1) // block_k
            latency_mainloop = (latency_load_per_stage + latency_smem_use) * num_kblocks
        else:
            gmem_load_prologue = gmem_load_per_stage * (stages - 1)
            # first load
            smem_load_per_tile = (
                block_m * k_tile * adtype.nbytes
                + block_n * k_tile * bdtype.nbytes
            )
            latency_prologue = (
                gmem_load_prologue / load_bw
                + gmem_load_prologue / smem_gbps_per_cta
                + smem_load_per_tile / smem_gbps_per_cta
            ) / 1e9
            compute_simd_per_tile = 2 * block_n * k_tile
            compute_tc_per_tile = 2 * block_m * block_n * k_tile
            latency_simd_per_tile = compute_simd_per_tile / simd_tput_per_cta / 1e12
            latency_tc_per_tile = compute_tc_per_tile / tc_tput_per_cta / 1e12
            # rough estimation
            num_tiles = block_k // k_tile
            latency_compute_per_tile = latency_tc_per_tile + latency_simd_per_tile
            smem_load_per_tile = (
                block_m * k_tile * adtype.nbytes
                + block_n * k_tile * bdtype.nbytes
            )
            latency_smem_per_tile = smem_load_per_tile / smem_gbps_per_cta / 1e9
            if latency_smem_per_tile <= latency_compute_per_tile:
                latency_smem_use = latency_smem_per_tile + latency_compute_per_tile * num_tiles
            else:
                latency_smem_use = latency_smem_per_tile * num_tiles + latency_compute_per_tile

            latency_load_per_stage = (gmem_load_per_stage / load_bw + gmem_load_per_stage / smem_gbps_per_cta) / 1e9
            parallel_k_parts = config.parallel_k_parts
            k_partition = block_k
            while k_partition * parallel_k_parts < k:
                k_partition += block_k
            num_kblocks = (k_partition + block_k - 1) // block_k
            if latency_load_per_stage <= latency_smem_use:
                latency_mainloop = latency_smem_use * num_kblocks
            else:
                latency_mainloop = latency_load_per_stage * (num_kblocks - stages + 1) + latency_smem_use * (stages - 1)
        
        smem_load_epilogue = smem_store_epilogue = gmem_store_epilogue = block_m * block_n * f16.nbytes
        # rough estimation
        store_bw = dram_gbps_per_cta * 0.6
        if parallel_k_parts == 1:
            latency_epilogue = (
                smem_load_epilogue + smem_store_epilogue
            ) / smem_gbps_per_cta + gmem_store_epilogue / store_bw
        else:
            # assume 80% of (following) loads are in L2 cache
            gmem_load = gmem_store_epilogue * (parallel_k_parts - 1)
            gmem_load_dram = gmem_load * 0.2
            gmem_load_l2 = gmem_load * 0.8
            latency_epilogue = (
                (smem_load_epilogue + smem_store_epilogue) / smem_gbps_per_cta
                + gmem_load_dram / load_bw
                + gmem_load_l2 / l2_gbps_per_cta
                + gmem_load / store_bw
            )
        latency_epilogue *= 1e-9

        latency_per_cta = latency_prologue + latency_mainloop + latency_epilogue
        latency_kernel = latency_per_cta * num_waves
        return latency_kernel * 1e6
