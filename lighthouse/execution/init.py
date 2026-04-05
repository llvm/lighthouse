"""
Infrastructure to initialize kenel arguments
"""

import numpy as np


class KernelArgument:
    """
    A kernel argument.
    Initialization is done at construction time.
    Accepted shape format is "DIMSxTYPExINIT" where:
    * DIMS: MxNxKx... (any number of dimensions)
    * TYPE: f16, f32, f64, bf16, i8, i16, i32, i64
    * INIT: 0, 1, rnd, id (identity)
    Access the argument value with the `argument` attribute, which is a numpy array.
    """

    def __init__(self, shape: str):
        shape, element_type, init_type = self._parse_shape(shape)
        self.argument = self._initialize(shape, init_type, element_type)

    def _parse_shape(self, shape_str: str) -> list[int]:
        """
        Parse a shape string in the format MxNx...xType into a list of integers.
        """
        init_type = shape_str.split("x")[-1]
        element_type_str = shape_str.split("x")[-2]
        element_type = self._get_input_type(element_type_str)
        try:
            dims = [int(dim) for dim in shape_str.split("x")[:-2]]
        except ValueError:
            raise ValueError(f"Invalid shape string: {shape_str}")
        return dims, element_type, init_type

    def _initialize(
        self, shape: tuple[int], init_type: str, element_type
    ) -> np.ndarray:
        """
        A simple initializer for kernel arguments.
        """
        if init_type == "0":
            return np.zeros(shape, dtype=element_type)
        elif init_type == "1":
            return np.ones(shape, dtype=element_type)
        elif init_type == "rnd":
            return np.random.rand(*shape).astype(element_type)
        elif init_type == "id":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError("Identity initializer only works for square matrices.")
            return np.eye(shape[0], dtype=element_type)
        else:
            raise ValueError(f"Unsupported initializer type: {init_type}")

    def _get_input_type(self, input_type_str: str):
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
