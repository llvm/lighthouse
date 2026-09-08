# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Cervený

"""Compile-time space-filling curve generators."""


def gilbert2d(width: int, height: int):
    """Generate a generalized Hilbert traversal of a 2D rectangle."""
    if width <= 0 or height <= 0:
        raise ValueError("SFC dimensions must be positive")
    if width >= height:
        yield from _generate2d(0, 0, width, 0, 0, height)
    else:
        yield from _generate2d(0, 0, 0, height, width, 0)


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def _generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int):
    width = abs(ax + ay)
    height = abs(bx + by)
    dax, day = _sign(ax), _sign(ay)
    dbx, dby = _sign(bx), _sign(by)

    if height == 1:
        for _ in range(width):
            yield x, y
            x, y = x + dax, y + day
        return

    if width == 1:
        for _ in range(height):
            yield x, y
            x, y = x + dbx, y + dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    width2 = abs(ax2 + ay2)
    height2 = abs(bx2 + by2)

    if 2 * width > 3 * height:
        if width2 % 2 and width > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _generate2d(x, y, ax2, ay2, bx, by)
        yield from _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        return

    if height2 % 2 and height > 2:
        bx2, by2 = bx2 + dbx, by2 + dby
    yield from _generate2d(x, y, bx2, by2, ax2, ay2)
    yield from _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
    yield from _generate2d(
        x + (ax - dax) + (bx2 - dbx),
        y + (ay - day) + (by2 - dby),
        -bx2,
        -by2,
        -(ax - ax2),
        -(ay - ay2),
    )
