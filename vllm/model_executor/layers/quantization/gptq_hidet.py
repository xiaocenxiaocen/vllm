# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from typing import Any, Dict, List, Optional

from fractions import Fraction
import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, LinearKernelBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_empty_g_idx
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from .weight_utils import reconstruct_21bit, weight_quantization_subbyte
from .hidet_w234_kernel import w2a16_linear, w3a16_linear, w4a16_linear
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.scalar_type import scalar_types


logger = init_logger(__name__)


class GPTQHidetConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    TYPE_MAP = {
        (3, True): scalar_types.uint4,
        (4, True): scalar_types.uint4,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
    ) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.pack_factor = Fraction(32, weight_bits)  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config: "
                             f"bits={weight_bits}, sym={is_sym}")

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (f"GPTQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_hidet"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQHidetConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, is_sym,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_gptq_hidet_compatible(hf_quant_cfg)

        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "gptq_marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_hidet"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_hidet for"
                        " faster inference")
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["GPTQHidetLinearMethod"]:
        if isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead)
                                             and self.lm_head_quantized):
            return GPTQHidetLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_gptq_hidet_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        sym = quant_config.get("sym", None)
        desc_act = quant_config.get("desc_act", None)

        if desc_act:
            return False

        if quant_method != "gptq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or sym is None
                or desc_act is None):
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False
        return True


class GPTQLinearKernel(LinearKernelBase):
    """Linear kernel for AWQ
    """
    def __init__(self, input_size: int, output_size: int, group_size: int, bit_width: int = 4):
        if bit_width == 4:
            self.linear_kernel = w4a16_linear("quant", input_size, output_size, group_size)
        else:
            assert bit_width == 3
            self.linear_kernel = w3a16_linear("quant", input_size, output_size, group_size)

    def apply_tensors(self, *tensors: torch.Tensor):
        return self.linear_kernel(*tensors)


class GPTQHidetLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Hidet.

    Args:
        quant_config: The GPTQ Hidet quantization config.
    """

    def __init__(self, quant_config: GPTQHidetConfig) -> None:
        self.quant_config = quant_config
        self.weight_bits = quant_config.weight_bits

    def create_linear_kernel(
        self,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype
    ) -> LinearKernelBase:
        return GPTQLinearKernel(input_size_per_partition, sum(output_partition_sizes), self.quant_config.group_size, self.weight_bits)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:

        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")
        is_channelwise = self.quant_config.group_size == -1

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if is_channelwise and is_row_parallel:
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        # Activation order
        g_idx = RowvLLMParameter(data=torch.empty(
            input_size_per_partition,
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)

        qzeros_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size

    # Checkpoints are serialized in AutoGPTQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking, including the activation reordering case.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device

        # required by torch.compile
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # Handle sorting for activation reordering if needed.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.zp = marlin_make_empty_g_idx(device)

        if self.weight_bits == 3:
            w2, w1 = reconstruct_21bit(layer.qweight)
            hidet_qweight2 = weight_quantization_subbyte(w2, 2)
            hidet_qweight1 = weight_quantization_subbyte(w1, 1)
            del layer.qweight
            del w2
            del w1
            layer.qweight2 = Parameter(hidet_qweight2, requires_grad=False)
            layer.qweight1 = Parameter(hidet_qweight1, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight_bits == 3:
            qweight2 = layer.qweight2
            qweight1 = layer.qweight1
            scales = layer.scales
            pack_factor = 16
            out_shape = x.shape[:-1] + (qweight2.shape[-1] * pack_factor,)
            reshaped_x = x.reshape(-1, x.shape[-1])

            assert layer.linear_kernel is not None
            out = layer.linear_kernel.apply_tensors(reshaped_x, qweight2, qweight1, scales)
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)
        qweight = layer.qweight
        scales = layer.scales
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        assert layer.linear_kernel is not None
        out = layer.linear_kernel.apply_tensors(reshaped_x, qweight, scales)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
