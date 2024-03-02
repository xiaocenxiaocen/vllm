# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import torch
import hidet

from hidet.lang import attrs
from hidet.lang.cuda import (
    blockIdx,
)
from hidet.ir.type import DataType

from hidet.ir.cute.layout import TensorLayout, make_layout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    rearrange,
    arithmetic,
    cast,
)

from hidet.ir.cute import auto_layout
from hidet.ir.cute import composition, coalesce, logical_divide

from hidet.lang.types import f32, bf16, f16, u4
from hidet.utils.py import cdiv


def wrapper_func(func, *tensors: torch.Tensor):
    s1 = hidet.cuda.current_stream().handle()
    s2 = torch.cuda.current_stream().cuda_stream
    if s1 == s2:
        func(*tensors)
    else:
        s = hidet.cuda.ExternalStream(s2)
        with hidet.cuda.stream(s):
            func(*tensors)


def depreprocess_weight(weight: torch.Tensor):
    m, n = weight.shape
    dtype = weight.dtype
    element_size = weight.element_size()
    pack_factor = element_size * 8 // u4.nbits
    n = n * pack_factor
    w = torch.empty(m, n // pack_factor, dtype=dtype, device="cuda")
    bm, bn = 64, 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    weight = weight.cuda()
    basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (m // n_mode.size(),)
    n_stride = n_mode.stride + (basic_block.cosize(),)
    n_mode = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (n // m_mode.size(),)
    cosize = m // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize,)
    m_mode = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(n_mode, m_mode)

    layout = TensorLayout((m, n))
    tile = TensorLayout((bm, bn), (1, m))
    tile = logical_divide(layout, tile)
    tile = composition(gmem_layout, tile)
    gmem, strides = tile
    m_stride, n_stride = strides.stride
    m_stride //= bm
    n_stride //= bn * m

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[n, m], wo: u4[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wi = tensor_view(wi[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi)

            tr_w = cast(tr_wi, f16)
            tr_w_cvt = rearrange(tr_w, auto_layout, "register")
            tr_wo = cast(tr_w_cvt, u4)

            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo)

    func = script_module.build()
    wrapper_func(func, weight, w)
    return w


def cast_u4_to_f16_interleaved(t: torch.Tensor):
    m, n = t.shape
    n = n * t.element_size() * 8 // u4.nbits
    ts = torch.empty(m, n, dtype=torch.float16, device="cuda")
    bm, bn = 64, 64
    threads = 128

    from hidet.ir.primitives.cuda.cvt import cast_u4x8_to_f16x8_interleaved_func

    cast_interleaved = cast_u4x8_to_f16x8_interleaved_func()

    t = t.cuda()
    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = arithmetic(tr_wi, op=cast_interleaved)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def cast_u4_to_f16(t: torch.Tensor):
    m, n = t.shape
    n = n * t.element_size() * 8 // u4.nbits
    ts = torch.empty(m, n, dtype=torch.float16, device="cuda")
    bm, bn = 64, 64
    threads = 128

    t = t.cuda()
    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = cast(tr_wi, f16)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def cast_f16_to_u4(t: torch.Tensor):
    m, n = t.shape
    ts = torch.empty(m, n // 8, dtype=torch.int32, device="cuda")
    bm, bn = 64, 64
    threads = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: f16[m, n], wo: u4[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(f16, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = cast(tr_wi, u4)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def preprocess_weight(weight: torch.Tensor):
    m, n = weight.shape
    dtype = weight.dtype
    element_size = weight.element_size()
    pack_factor = element_size * 8 // u4.nbits
    n = n * pack_factor
    w = torch.empty(m, n // pack_factor, dtype=dtype, device="cuda")
    bm, bn = 64, 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    if not weight.is_contiguous():
        weight = weight.contiguous()
    basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (m // n_mode.size(),)
    n_stride = n_mode.stride + (basic_block.cosize(),)
    n_mode = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (n // m_mode.size(),)
    cosize = m // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize,)
    m_mode = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(n_mode, m_mode)

    layout = TensorLayout((m, n))
    tile = TensorLayout((bm, bn), (1, m))
    tile = logical_divide(layout, tile)
    tile = composition(gmem_layout, tile)
    gmem, strides = tile
    m_stride, n_stride = strides.stride
    m_stride //= bm
    n_stride //= bn * m

    from hidet.ir.primitives.cuda.cvt import cast_u4x8_to_f16x8_interleaved_func

    cast_interleaved = cast_u4x8_to_f16x8_interleaved_func()

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: u4[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi)

            tr_w = arithmetic(tr_wi, op=cast_interleaved)
            tr_w_cvt = rearrange(tr_w, auto_layout, "register")
            tr_wo = cast(tr_w_cvt, u4)

            tg_wo = tensor_view(wo[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo)

    func = script_module.build()
    wrapper_func(func, weight, w)
    return w


def bench(f, warmup=1, iter=20):
    import time

    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)
    return res * 1000.0


def canonicalize(layout: TensorLayout):
    return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))


def hidet_dtype_to_torch_dtype(dtype: DataType):
    if dtype == f32:
        return torch.float32
    elif dtype == f16:
        return torch.float16
    elif dtype == bf16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")