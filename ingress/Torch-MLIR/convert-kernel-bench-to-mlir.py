#!/usr/bin/env python3

import importlib
import importlib.util

from pathlib import Path

from mlir import ir, passmanager
from torch_mlir import fx

kernels_as_pytorch_folder = Path(__file__).parent / "KernelBench" / "KernelBench"
kernels_as_pytorch_level1 = kernels_as_pytorch_folder / "level1"
kernels_as_pytorch_level2 = kernels_as_pytorch_folder / "level2"

kernels_as_mlir_folder = Path(__file__).parent / "cache"
kernels_as_mlir_level1 = kernels_as_mlir_folder / "level1"
kernels_as_mlir_level1.mkdir(parents=True, exist_ok=True)
kernels_as_mlir_level2 = kernels_as_mlir_folder / "level2"
kernels_as_mlir_level2.mkdir(parents=True, exist_ok=True)

level1, level2 = Path("level1"), Path("level2")
ignore_list = [
    level1 / "12_Matmul_with_diagonal_matrices_.py",  # torch.operator "torch.aten.diag"
    level1
    / "34_InstanceNorm.py",  # LLVM ERROR: SmallVector unable to grow. Requested capacity (93898875033000)
    level1
    / "72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_.py",  # Bare exception during torch-backend-to-linalg-on-tensors-backend-pipeline
    level1
    / "89_cumsum.py",  # Dialect `tm_tensor' not found for custom op 'tm_tensor.scan'
    level1
    / "90_cumprod.py",  # Dialect `tm_tensor' not found for custom op 'tm_tensor.scan'
    level1
    / "91_cumsum_reverse.py",  # Dialect `tm_tensor' not found for custom op 'tm_tensor.scan'
    level1
    / "92_cumsum_exclusive.py",  # Dialect `tm_tensor' not found for custom op 'tm_tensor.scan'
    level1
    / "93_masked_cumsum.py",  # Dialect `tm_tensor' not found for custom op 'tm_tensor.scan'
    level1
    / "95_CrossEntropyLoss.py",  # Bare exception during torch-backend-to-linalg-on-tensors-backend-pipeline
    level1
    / "96_HuberLoss.py",  # Bare exception during torch-backend-to-linalg-on-tensors-backend-pipeline
    level1
    / "97_ScaledDotProductAttention.py",  # AssertionError: Torch not compiled with CUDA enabled
    level1
    / "99_TripletMarginLoss.py",  # Bare exception during torch-backend-to-linalg-on-tensors-backend-pipeline
    level2
    / "17_Conv2d_InstanceNorm_Divide.py",  # LLVM ERROR: SmallVector unable to grow. Requested capacity (94899412484104)
    level2
    / "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",  # LLVM ERROR: SmallVector unable to grow. Requested capacity (94899412484104)
    level2
    / "42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "43_Conv3d_Max_LogSumExp_ReLU.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "45_Gemm_Sigmoid_LogSumExp.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "52_Conv2d_Activation_BatchNorm.py",  # failed to legalize operation 'torch.operator'
    level2 / "55_Matmul_MaxPool_Sum_Scale.py",  # MLIR file too big: 16G
    level2 / "59_Matmul_Swish_Scaling.py",  # MLIR file too big: 16G
    level2 / "56_Matmul_Sigmoid_Sum.py",  # MLIR file too big: 16G
    level2 / "66_Matmul_Dropout_Softmax.py",  # MLIR file too big: 4G
    level2 / "68_Matmul_Min_Subtract.py",  # MLIR file too big: 4G
    level2 / "94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.py",  # MLIR file too big: 1G
    level2 / "33_Gemm_Scale_BatchNorm.py",  # MLIR file too big: 1G
    level2 / "88_Gemm_GroupNorm_Swish_Multiply_Swish.py",  # MLIR file too big: 1G
    level2 / "75_Gemm_GroupNorm_Min_BiasAdd.py",  # MLIR file too big: 1G
    level2 / "84_Gemm_BatchNorm_Scaling_Softmax.py",  # MLIR file too big: 1G
    level2 / "97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py",  # MLIR file too big: 1G
    level2 / "62_Matmul_GroupNorm_LeakyReLU_Sum.py",  # MLIR file too big: 1G
    level2 / "30_Gemm_GroupNorm_Hardtanh.py",  # MLIR file too big: 1G
    level2 / "95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py",  # MLIR file too big: 1G
    level2 / "29_Matmul_Mish_Mish.py",  # MLIR file too big: 1G
    level2 / "99_Matmul_GELU_Softmax.py",  # MLIR file too big: 1G
    level2 / "98_Matmul_AvgPool_GELU_Scale_Max.py",  # MLIR file too big: 1G
    level2 / "80_Gemm_Max_Subtract_GELU.py",  # MLIR file too big: 1G
    level2 / "81_Gemm_Swish_Divide_Clamp_Tanh_Clamp.py",  # MLIR file too big: 1G
    level2 / "12_Gemm_Multiply_LeakyReLU.py",  # MLIR file too big: 1G
    level2 / "53_Gemm_Scaling_Hardtanh_GELU.py",  # MLIR file too big: 1G
    level2 / "9_Matmul_Subtract_Multiply_ReLU.py",  # MLIR file too big: 1G
    level2 / "70_Gemm_Sigmoid_Scaling_ResidualAdd.py",  # MLIR file too big: 1G
    level2 / "86_Matmul_Divide_GELU.py",  # MLIR file too big: 1G
    level2 / "63_Gemm_ReLU_Divide.py",  # MLIR file too big: 1G
    level2 / "76_Gemm_Add_ReLU.py",  # MLIR file too big: 1G
    level2 / "14_Gemm_Divide_Sum_Scaling.py",  # MLIR file too big: 1G
    level2 / "39_Gemm_Scale_BatchNorm.py",  # MLIR file too big: 256M
    level2 / "41_Gemm_BatchNorm_GELU_ReLU.py",  # MLIR file too big: 256M
    level2 / "40_Matmul_Scaling_ResidualAdd.py",  # MLIR file too big: 256M
    level2 / "37_Matmul_Swish_Sum_GroupNorm.py",  # MLIR file too big: 64.3M
    level2
    / "58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py",  # error: failed to legalize operation 'torch.constant.int'
    level2
    / "79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max.py",  # LLVM ERROR: SmallVector unable to grow. Requested capacity (94312016449768)
    level2
    / "92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp.py",  # error: failed to legalize operation 'torch.constant.int'
]


ctx = ir.Context()
pm = passmanager.PassManager(context=ctx)
pm.add("linalg-specialize-generic-ops")

for pytorch_level, mlir_level in (
    (kernels_as_pytorch_level1, kernels_as_mlir_level1),
    (kernels_as_pytorch_level2, kernels_as_mlir_level2),
):
    for kernel_pytorch_file in pytorch_level.iterdir():
        level_and_kernel = (
            Path(kernel_pytorch_file.parent.name) / kernel_pytorch_file.name
        )
        if level_and_kernel in ignore_list or not kernel_pytorch_file.is_file():
            print(
                f"Skipping: {kernel_pytorch_file.parent.name}/{kernel_pytorch_file.name}"
            )
            continue

        module_name = kernel_pytorch_file.stem

        kernel_as_mlir_path = mlir_level / (module_name + ".mlir")
        if kernel_as_mlir_path.exists():
            print(
                f"Already in cache: {kernel_pytorch_file.parent.name}/{kernel_pytorch_file.name}"
            )
            continue
        print(
            f"Processing: {kernel_pytorch_file.parent.name}/{kernel_pytorch_file.name}"
        )

        module_spec = importlib.util.spec_from_file_location(
            module_name, kernel_pytorch_file
        )

        if module_spec is None or module_spec.loader is None:
            print(f"Error: Could not create module spec for {kernel_pytorch_file}")
            continue

        module = importlib.util.module_from_spec(module_spec)
        # Execute the module to load its contents
        module_spec.loader.exec_module(module)

        if not all(
            hasattr(module, a) for a in ("Model", "get_inputs", "get_init_inputs")
        ):
            print(f"Error: module in file {kernel_pytorch_file} not a proper benchmark")
            continue

        # TODO: check hasattr(module, "in_features") etc and adjust to sizes that are more tractable for torch-mlir

        try:
            m = fx.export_and_import(
                module.Model(*module.get_init_inputs()),
                *module.get_inputs(),
                output_type=fx.OutputType.LINALG_ON_TENSORS,
            )
        except Exception as e:
            print(f"Error: got the following error converting {kernel_pytorch_file}")
            raise e

        before_clean_up = "//" + str(m)[:-1].replace("\n", "\n//") + "\n"
        # Cross boundary from torch-mlir's mlir to environment's mlir
        m = ir.Module.parse(str(m), context=ctx)
        # Run clean-up, e.g. linalg-"specialization" passes to raise within Linalg.
        try:
            pm.run(m.operation)  # cleanup
        except Exception as e:
            print(f"Error: got the following error cleaning up {module_name}")
            raise e

        with kernel_as_mlir_path.open("w") as f:
            print("// Torch-MLIR output:", file=f)
            print(before_clean_up, file=f)
            print("// MLIR output after clean-up:", file=f)
            print(m, file=f)
