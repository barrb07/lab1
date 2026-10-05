"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros_like(vectors[0], dtype=float)
    for m, v in zip(matrices, vectors):
        result += m @ v
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    np.random.seed(seed)
    mat = np.random.normal(mean, std, (rows, columns))
    return MatrixStatistics(
        matrix=mat,
        row_means=np.mean(mat, axis=1).tolist(),
        col_means=np.mean(mat, axis=0).tolist(),
        row_vars=np.var(mat, axis=1).tolist(),
        col_vars=np.var(mat, axis=0).tolist()
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices = np.arange(rows).reshape(-1, 1)
    col_indices = np.arange(columns).reshape(1, -1)
    mask = (row_indices + col_indices) % 2 == 0
    return np.where(mask, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    start_y = (image_height - height) // 2
    start_x = (image_width - width) // 2
    img[start_y:start_y + height, start_x:start_x + width] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    y0, x0 = image_height // 2, image_width // 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    series = np.array(values, dtype=float)
    
    mean = float(np.mean(series))
    variance = float(np.var(series))
    std = float(np.std(series))
    
    max_mask = (series[1:-1] > series[:-2]) & (series[1:-1] > series[2:])
    local_max_indices = (np.where(max_mask)[0] + 1).tolist()
    
    min_mask = (series[1:-1] < series[:-2]) & (series[1:-1] < series[2:])
    local_min_indices = (np.where(min_mask)[0] + 1).tolist()
    
    n = len(series)
    moving_average = np.array([np.mean(series[i:i + window]) for i in range(n - window + 1)]).tolist()
    
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_max_indices=local_max_indices,
        local_min_indices=local_min_indices,
        moving_average=moving_average
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels_arr = np.array(labels, dtype=int)
    if class_count is None:
        class_count = int(np.max(labels_arr)) + 1
    n = len(labels_arr)
    result = np.zeros((n, class_count), dtype=int)
    result[np.arange(n), labels_arr] = 1
    return result
