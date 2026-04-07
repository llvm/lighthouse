"""
Infrastructure to initialize kenel arguments
"""

from enum import Enum

import numpy as np


class InitType(Enum):
    """Initialization type for kernel arguments.
    * ZERO: initialize all elements to zero
    * ONE: initialize all elements to one
    * RANDOM: initialize all elements to random values (needs extra config for distribution)
    * IDENTITY: initialize to identity matrix (only for 2D square matrices)
    """

    ZERO = "0"
    ONE = "1"
    RANDOM = "rnd"
    IDENTITY = "id"


class KernelArgument:
    """
    A kernel argument, initialized according to the specified type.
    The argument value is stored in the `arg` attribute, which is a numpy array.
    It will be initialized at construction time, so that the argument value is ready to use after construction.

    Arguments are:
    * dims: list of dimensions of the argument > 0 (e.g., [M, N, K])
    * element_type: NumPy data type of the argument (e.g., np.float32, np.int64, "f16", "bf16", etc.)
    * init_type: type of initialization (InitType)
    TODO: Add support for distribution parameters on random.
    """

    dims: list[int]
    element_type: np.dtype
    init_type: InitType
    arg: np.ndarray

    def __init__(
        self,
        dims: list[int],
        element_type: np.dtype,
        init_type: InitType = InitType.ZERO,
    ):
        if (
            dims is None
            or len(dims) == 0
            or not all(isinstance(dim, int) and dim > 0 for dim in dims)
        ):
            raise ValueError(
                "Dimensions must be a non-empty list of non-zero integers."
            )
        self.dims = dims
        self.element_type = element_type
        self.init_type = init_type
        self.arg = self.initialize(dims, init_type, element_type)

    def initialize(
        self, shape: list[int], init_type: InitType, element_type: np.dtype
    ) -> np.ndarray:
        """
        A simple initializer for kernel arguments.
        """
        if init_type == InitType.ZERO:
            return np.zeros(shape, dtype=element_type)
        elif init_type == InitType.ONE:
            return np.ones(shape, dtype=element_type)
        elif init_type == InitType.RANDOM:
            return np.random.rand(*shape).astype(element_type)
        elif init_type == InitType.IDENTITY:
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("Identity initializer only works for square matrices.")
            return np.eye(shape[0], dtype=element_type)
        else:
            raise ValueError(f"Unsupported initializer type: {init_type}")


class KernelArgumentParser:
    """
    A kernel CLI argument parser.
    Accepted shape format is "DIMSxTYPExINIT" where:
    * DIMS: MxNxKx... (any number of dimensions >= 1)
    * TYPE: f16, f32, f64, bf16, i8, i16, i32, i64
    * INIT: InitType (0, 1, rnd, id)
    Returns a KernelArgument.
    """

    @staticmethod
    def parse(shape: str):
        return KernelArgument(*KernelArgumentParser._parse_shape(shape))

    @staticmethod
    def _parse_shape(shape_str: str) -> list[int]:
        """
        Parse a shape string in the format MxNx...xTypexInit into a list of integers.
        """
        element_type = KernelArgumentParser._get_input_type(shape_str.split("x")[-2])
        try:
            dims = [int(dim) for dim in shape_str.split("x")[:-2]]
            init_type = InitType(shape_str.split("x")[-1])
        except ValueError:
            raise ValueError(f"Invalid shape string: {shape_str}")
        return (dims, element_type, init_type)

    @staticmethod
    def _get_input_type(input_type_str: str):
        if input_type_str == "f16":
            return np.float16
        elif input_type_str == "bf16":
            return np.bfloat16
        elif input_type_str == "f32":
            return np.float32
        elif input_type_str == "f64":
            return np.float64

        elif input_type_str == "i8":
            return np.int8
        elif input_type_str == "i16":
            return np.int16
        elif input_type_str == "i32":
            return np.int32
        elif input_type_str == "i64":
            return np.int64
        else:
            raise ValueError(f"Unsupported input type: {input_type_str}")
