import ctypes
import ctypes.util

SYS_ARCH_PRCTL = 158
ARCH_REQ_XCOMP_PERM = 0x1023
XFEATURE_XTILEDATA = 18


def enable_amx():
    """
    Enable AMX support on the current thread.

    Raises:
        OSError: If the syscall fails
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    ret = libc.syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
    if ret != 0:
        raise OSError(
            ctypes.get_errno(),
            "arch_prctl failed: AMX not supported or permission denied",
        )
