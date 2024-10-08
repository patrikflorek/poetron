"""
This module provides utility functions for calculating some statistical measures.
"""


def calculate_average(data):
    """
    Calculate the average of the given data.

    Args:
        data (list): List of numbers.

    Returns:
        float: Average of the data.
    """
    return sum(data) / len(data)


def calculate_median(data):
    """
    Calculate the median of the given data.

    Args:
        data (list): List of numbers.

    Returns:
        float: Median of the data.
    """
    data.sort()
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[n // 2]


def calculate_mode(data):
    """
    Calculate the mode of the given data.

    Args:
        data (list): List of numbers.

    Returns:
        float: Mode of the data.
    """
    return max(set(data), key=data.count)


def calculate_std_dev(data):
    """
    Calculate the standard deviation of the given data.

    Args:
        data (list): List of numbers.

    Returns:
        float: Standard deviation of the data.
    """
    mean = calculate_average(data)
    return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
