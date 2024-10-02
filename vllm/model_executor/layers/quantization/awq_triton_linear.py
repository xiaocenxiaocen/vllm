# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import Any, Dict, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, LinearKernelBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.model_executor.layers.quantization.utils.marlin_utils import replace_tensor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from .awq_triton import awq_gemm_triton, AWQ_TRITON_SUPPORTED_GROUP_SIZES


logger = init_logger(__name__)


class AWQTritonConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(self, weight_bits: int, group_size: int, zero_point: bool, lm_head_quantized: bool) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for " f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQHidetConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )

    @classmethod
    def get_name(self) -> str:
        return "awq_triton"

    @classmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQTritonConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, zero_point, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_awq_triton_compatible(hf_quant_cfg)
        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "awq_marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info("Detected that the model can run with awq_triton"
                        ", however you specified quantization=awq explicitly,"
                        " so forcing awq. Use quantization=awq_hidet for"
                        " faster inference")
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["AWQTritonLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return AWQTritonLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_awq_triton_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        has_zp = quant_config.get("zero_point", None)

        if quant_method != "awq":
            return False

        if num_bits != 4 or group_size not in AWQ_TRITON_SUPPORTED_GROUP_SIZES or has_zp is None:
            return False
        return True


class AWQTritonLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQTritonConfig):
        self.quant_config = quant_config

    def create_linear_kernel(
        self,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype
    ) -> LinearKernelBase:
        return None

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        #print("after loading")
        #print("qw", layer.qweight.shape, layer.qweight.dtype)
        #print("zp", layer.qzeros.shape, layer.qzeros.dtype)
        #print("scales", layer.scales.shape, layer.scales.dtype)
        layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        m, _ = reshaped_x.shape
        if m >= 128:
            out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, 1, 128, 128, 32)
        elif m >= 64:
            out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, 1, 64, 64, 32)
        elif m >= 32:
            out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, 4, 32, 128, 64)
        else:
            out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, 8, 16, 128, 64)
        return out.reshape(out_shape)