import numpy as np


def largest_inscribed_rectangle(segmentation_map, label):
    """
    Find the largest inscribed rectangle inside a specific segmentation region,
    ensuring that the rectangle's width is larger than its height.

    Parameters:
    segmentation_map (np.ndarray): A 2D numpy array representing the segmentation map.
    label (int): The label of the segmentation region for which the rectangle is to be inscribed.

    Returns:
    (x, y, w, h): The top-left corner (x, y) and dimensions (w, h) of the largest inscribed rectangle.
    """
    # Create a binary mask for the specific label
    binary_map = (segmentation_map == label).astype(np.int32)

    # Dimensions of the segmentation map
    rows, cols = binary_map.shape

    # DP tables for the largest width and height ending at each point (i, j)
    width = np.zeros_like(binary_map)
    height = np.zeros_like(binary_map)

    # Variables to keep track of the largest rectangle
    max_area = 0
    best_rectangle = (0, 0, 0, 0)  # x, y, width, height

    # Traverse the grid
    for i in range(rows):
        for j in range(cols):
            if binary_map[i, j] == 1:
                # Update height and width for (i, j)
                height[i, j] = height[i - 1, j] + 1 if i > 0 else 1
                width[i, j] = width[i, j - 1] + 1 if j > 0 else 1

                # Check for the largest rectangle ending at (i, j)
                min_width = width[i, j]
                for k in range(height[i, j]):
                    min_width = min(min_width, width[i - k, j])
                    rect_height = k + 1
                    rect_width = min_width

                    # Only consider rectangles where width > height
                    if rect_width > rect_height:
                        area = rect_width * rect_height
                        if area > max_area:
                            max_area = area
                            best_rectangle = (j - rect_width + 1, i - rect_height + 1, rect_width, rect_height)

    return best_rectangle


def largest_inscribed_rectangle_faster(segmentation_map, label):
    """
    Find the largest inscribed rectangle inside a specific segmentation region,
    ensuring that the rectangle's width is larger than its height.

    Parameters:
    segmentation_map (np.ndarray): A 2D numpy array representing the segmentation map.
    label (int): The label of the segmentation region for which the rectangle is to be inscribed.

    Returns:
    (x, y, w, h): The top-left corner (x, y) and dimensions (w, h) of the largest inscribed rectangle.
    """
    # Create a binary mask for the specific label
    binary_map = (segmentation_map == label).astype(np.int32)

    # Dimensions of the segmentation map
    rows, cols = binary_map.shape

    # DP tables for the largest width and height ending at each point (i, j)
    height = np.zeros_like(binary_map)
    max_area = 0
    best_rectangle = (0, 0, 0, 0)  # x, y, width, height

    for i in range(rows):
        width_stack = []
        for j in range(cols):
            # Update the height map
            height[i, j] = height[i - 1, j] + 1 if binary_map[i, j] == 1 and i > 0 else binary_map[i, j]

            # Process the current row as a histogram
            start = j
            while width_stack and height[i, width_stack[-1]] > height[i, j]:
                h = height[i, width_stack.pop()]
                w = j - width_stack[-1] - 1 if width_stack else j
                if w > h:  # Only consider rectangles where width > height
                    area = w * h
                    if area > max_area:
                        max_area = area
                        best_rectangle = (width_stack[-1] + 1 if width_stack else 0, i - h + 1, w, h)
                start = width_stack[-1] + 1 if width_stack else 0

            width_stack.append(j)

        # Clear remaining elements in the stack
        while width_stack:
            h = height[i, width_stack.pop()]
            w = cols - width_stack[-1] - 1 if width_stack else cols
            if w > h:
                area = w * h
                if area > max_area:
                    max_area = area
                    best_rectangle = (width_stack[-1] + 1 if width_stack else 0, i - h + 1, w, h)

    return best_rectangle
