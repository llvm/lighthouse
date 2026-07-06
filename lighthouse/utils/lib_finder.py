import ctypes
import ctypes.util
from functools import cache


class _LinkMap(ctypes.Structure):
    """Wrapper around the 'link_map' structure from libdl."""

    _fields_ = [
        ("l_addr", ctypes.c_void_p),
        ("l_name", ctypes.c_char_p),
        ("l_ld", ctypes.c_void_p),
        ("l_next", ctypes.c_void_p),
        ("l_prev", ctypes.c_void_p),
    ]


class DLInfo:
    """
    Wrapper around 'libdl' to obtain information about a dynamically
    loaded objects.
    """

    def __init__(self):
        self.libdl = self.load_library("dl")
        if not self.libdl:
            raise RuntimeError("Could not load libdl.so")
        self.dlinfo = self.libdl.dlinfo
        self.dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.dlinfo.restype = ctypes.c_int

    @staticmethod
    def load_library(lib_name: str) -> ctypes.CDLL | None:
        """
        Load a dynamically linked library.

        Args:
            lib_name: Full or partial name of the library to load
                (e.g., "c", "libomp", "libm.so")
        Returns:
            Handle to the loaded library on success.
        """
        lib_path = ctypes.util.find_library(lib_name)
        if not lib_path:
            return None

        try:
            lib = ctypes.CDLL(lib_path)
        except OSError:
            return None

        return lib

    def lib_path(self, lib_name: str) -> str | None:
        """
        Get the full path of a shared library.

        Args:
            lib_name: Full or partial name of the library to find
                (e.g., "c", "libomp", "libm.so")
        Returns:
            Full path to the library if found, None otherwise.
        """
        lib = self.load_library(lib_name)
        if not lib:
            return None

        # Get the link map for the library.
        link_map = ctypes.POINTER(_LinkMap)()
        RTLD_DI_LINKMAP = 2
        if (
            self.dlinfo(lib._handle, RTLD_DI_LINKMAP, ctypes.byref(link_map)) != 0
            or not link_map
        ):
            return None

        # Get the full path.
        return link_map.contents.l_name.decode()


@cache
def find_openmp_library() -> str | None:
    """
    Locate an OpenMP runtime shared library on the host system.

    Returns:
        Absolute path if found, None otherwise.
    """
    try:
        dlinfo = DLInfo()
    except RuntimeError:
        return None
    return dlinfo.lib_path("omp")
