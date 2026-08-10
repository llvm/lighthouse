from collections.abc import Iterator
import json
import os


class ScheduleParameters:
    """
    Schedule parameters abstraction.

    ScheduleParameters is effectively a list of dictionaries, where each
    dictionary contains the parameters for a specific stage of the lowering
    schedule.

    Each dictionary must contain a key "layer_kind" that defines the kind of
    layer being transformed (e.g., "matmul"). The parameter values must be of
    type int, float, str, bool, or a list of thereof. The semantics of the
    parameters depend on the used lowering schedule.

    ScheduleParameters can be stored to/loaded from disk using the `to_json`
    and `from_json` methods.

    ScheduleParameters object can be used as a list of dictionaries:

    ```python
    params = ScheduleParameters(
        [
            {"layer_kind": "matmul", "wg_tile": [256, 256]},
        ]
    )
    for p in params:
        print(p["layer_kind"], p["wg_tile"])
    ```
    """

    def __init__(self, list_of_parameters: list[dict] | None):
        """
        Initialize the ScheduleParameters with an optional list of parameter dictionaries.
        """
        ScheduleParameters._verify_params_list(list_of_parameters)
        self.list_of_parameters = list_of_parameters or []

    def __iter__(self) -> Iterator[dict]:
        return iter(self.list_of_parameters)

    def __len__(self) -> int:
        return len(self.list_of_parameters)

    def __getitem__(self, index: int) -> dict:
        return self.list_of_parameters[index]

    def __setitem__(self, index: int, value: dict):
        self._verify_params_dict(value)
        self.list_of_parameters[index] = value

    def __delitem__(self, index: int):
        del self.list_of_parameters[index]

    def append(self, value: dict):
        self._verify_params_dict(value)
        self.list_of_parameters.append(value)

    def extend(self, values: list[dict]):
        for value in values:
            self._verify_params_dict(value)
        self.list_of_parameters.extend(values)

    def insert(self, index: int, value: dict):
        self._verify_params_dict(value)
        self.list_of_parameters.insert(index, value)

    @staticmethod
    def _verify_params_list(list_of_parameters: list[dict] | None):
        if list_of_parameters is None:
            return
        if not isinstance(list_of_parameters, list):
            raise ValueError(
                f"list_of_parameters must be a list of dictionaries or None, got {type(list_of_parameters)}"
            )
        for item in list_of_parameters:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Each item in list_of_parameters must be a dictionary, got {type(item)}"
                )
        for params in list_of_parameters:
            ScheduleParameters._verify_params_dict(params)

    @staticmethod
    def _verify_params_dict(params: dict):
        if not isinstance(params, dict):
            raise ValueError(
                f"Each parameter entry must be a dictionary: found '{params}'."
            )
        if "layer_kind" not in params:
            raise ValueError(
                "Each parameter dictionary must contain a 'layer_kind' key."
            )
        if not isinstance(params["layer_kind"], str):
            raise ValueError("'layer_kind' must be a string.")
        for key, value in params.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Parameter dictionary keys must be strings: found '{key}'."
                )
            if not isinstance(value, (int, float, str, bool, list, tuple)):
                raise ValueError(
                    f"Parameter dictionary values must be an int, float, str, bool, list, or tuple: found '{value}' for key '{key}'."
                )

    def to_json(self, filename, overwrite=False):
        """
        Serialize the schedule parameters to a JSON file.
        """
        if not overwrite and os.path.exists(filename):
            raise FileExistsError(
                f"File '{filename}' already exists. Use 'overwrite=True' to overwrite it."
            )
        with open(filename, "w") as f:
            json.dump(self.list_of_parameters, f, indent=4)

    @classmethod
    def from_json(cls, filename: str):
        """
        Deserialize the schedule parameters from a JSON file.
        """
        with open(filename) as f:
            list_of_parameters = json.load(f)
        instance = cls(list_of_parameters)
        return instance
