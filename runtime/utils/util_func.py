def max_below_threshold(lst, threshold):
    max_below = float('-inf')  # Initialize with negative infinity
    for num in lst:
        if num <= threshold and num > max_below:
            max_below = num
    return max_below