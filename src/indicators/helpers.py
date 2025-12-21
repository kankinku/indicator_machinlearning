"""
숫자 배열 유틸리티. indicatorts의 helper/numArray.ts를 Python으로 포팅.
간단한 리스트 기반 구현으로 NumPy 없이 동작하도록 작성했다.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


class IndicatorInputError(Exception):
    """입력 길이 불일치 등 계약 위반 시 발생."""


def check_same_length(*arrays: Sequence[float]) -> None:
    if not arrays:
        return
    length = len(arrays[0])
    for idx, arr in enumerate(arrays[1:], start=1):
        if len(arr) != length:
            raise IndicatorInputError(f"values length at {idx} not {length}")


def abs_values(values: Sequence[float]) -> List[float]:
    return [abs(v) for v in values]


def add(values1: Sequence[float], values2: Sequence[float]) -> List[float]:
    check_same_length(values1, values2)
    return [a + b for a, b in zip(values1, values2)]


def add_by(n: float, values: Sequence[float]) -> List[float]:
    return [v + n for v in values]


def subtract(values1: Sequence[float], values2: Sequence[float]) -> List[float]:
    check_same_length(values1, values2)
    return [a - b for a, b in zip(values1, values2)]


def subtract_by(n: float, values: Sequence[float]) -> List[float]:
    return [v - n for v in values]


def multiply(values1: Sequence[float], values2: Sequence[float]) -> List[float]:
    check_same_length(values1, values2)
    return [a * b for a, b in zip(values1, values2)]


def multiply_by(n: float, values: Sequence[float]) -> List[float]:
    return [v * n for v in values]


def divide(values1: Sequence[float], values2: Sequence[float]) -> List[float]:
    check_same_length(values1, values2)
    return [a / b for a, b in zip(values1, values2)]


def divide_by(n: float, values: Sequence[float]) -> List[float]:
    return [v / n for v in values]


def shift_right_and_fill_by(n: int, fill: float, values: Sequence[float]) -> List[float]:
    return [fill if i < n else values[i - n] for i in range(len(values))]


def shift_right_by(n: int, values: Sequence[float]) -> List[float]:
    return shift_right_and_fill_by(n, 0.0, values)


def shift_left_and_fill_by(n: int, fill: float, values: Sequence[float]) -> List[float]:
    length = len(values)
    result = [fill for _ in range(length)]
    for i in range(n, length):
        new_idx = (i - n + length) % length
        result[new_idx] = values[i]
    return result


def shift_left_by(n: int, values: Sequence[float]) -> List[float]:
    return shift_left_and_fill_by(n, 0.0, values)


def changes(n: int, values: Sequence[float]) -> List[float]:
    return subtract(values, shift_right_by(n, values))


def extract_signs(values: Sequence[float]) -> List[int]:
    return [1 if v >= 0 else -1 for v in values]


def transpose(*values: Sequence[float]) -> List[List[float]]:
    check_same_length(*values)
    return [list(row) for row in zip(*values)]


def max_rows(*values: Sequence[float]) -> List[float]:
    return [max(row) for row in transpose(*values)]


def round_digits(digits: int, value: float) -> float:
    n = 10 ** digits
    return round(value * n) / n


def round_digits_all(digits: int, values: Iterable[float]) -> List[float]:
    return [round_digits(digits, v) for v in values]


def pow_all(values: Sequence[float], exponent: float) -> List[float]:
    return [v**exponent for v in values]
