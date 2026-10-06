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


# ==========================================
# ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ: Гистограммы (Задача 4)
# ==========================================
def plot_histograms(stats: MatrixStatistics):
    """Построение гистограмм для строк и столбцов матрицы."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Гистограмма средних значений строк
    axes[0, 0].hist(stats.row_means, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Гистограмма средних значений строк')
    axes[0, 0].set_xlabel('Среднее значение')
    axes[0, 0].set_ylabel('Частота')
    
    # Гистограмма средних значений столбцов
    axes[0, 1].hist(stats.col_means, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Гистограмма средних значений столбцов')
    axes[0, 1].set_xlabel('Среднее значение')
    axes[0, 1].set_ylabel('Частота')
    
    # Гистограмма дисперсий строк
    axes[1, 0].hist(stats.row_vars, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Гистограмма дисперсий строк')
    axes[1, 0].set_xlabel('Дисперсия')
    axes[1, 0].set_ylabel('Частота')
    
    # Гистограмма дисперсий столбцов
    axes[1, 1].hist(stats.col_vars, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Гистограмма дисперсий столбцов')
    axes[1, 1].set_xlabel('Дисперсия')
    axes[1, 1].set_ylabel('Частота')
    
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Гистограммы сохранены в файл 'histograms.png'")


# ==========================================
# СОБСТВЕННЫЕ ТЕСТЫ
# ==========================================
if __name__ == "__main__":
    # Простые моки для тестирования
    class MockMatrixVectorBatchInput:
        def __init__(self, matrices, vectors):
            self.matrices = matrices
            self.vectors = vectors
    
    class MockBinarizeInput:
        def __init__(self, matrix, threshold):
            self.matrix = matrix
            self.threshold = threshold
    
    class MockMatrixInput:
        def __init__(self, matrix):
            self.matrix = matrix
    
    class MockRandomMatrixInput:
        def __init__(self, rows, columns, mean, std, seed):
            self.rows = rows
            self.columns = columns
            self.mean = mean
            self.std = std
            self.seed = seed
    
    class MockChessInput:
        def __init__(self, rows, columns, first, second):
            self.rows = rows
            self.columns = columns
            self.first = first
            self.second = second
    
    class MockRectangleInput:
        def __init__(self, width, height, image_height, image_width, shape_color, background_color):
            self.width = width
            self.height = height
            self.image_height = image_height
            self.image_width = image_width
            self.shape_color = shape_color
            self.background_color = background_color
    
    class MockEllipseInput:
        def __init__(self, semi_axis_x, semi_axis_y, image_height, image_width, shape_color, background_color):
            self.semi_axis_x = semi_axis_x
            self.semi_axis_y = semi_axis_y
            self.image_height = image_height
            self.image_width = image_width
            self.shape_color = shape_color
            self.background_color = background_color
    
    class MockTimeSeriesInput:
        def __init__(self, values, window):
            self.values = values
            self.window = window
    
    class MockOneHotInput:
        def __init__(self, labels, class_count):
            self.labels = labels
            self.class_count = class_count
    
    # Задача 1: Сумма произведений матриц на векторы
    print("Тесты для sum_prod:")
    m1 = np.array([[1, 0], [0, 1]])
    v1 = np.array([[1], [2]])
    result = sum_prod(MockMatrixVectorBatchInput([m1], [v1]))
    assert np.array_equal(result, np.array([[1.0], [2.0]])), "Единичная матрица"
    print("✅ Все тесты пройдены\n")
    
    # Задача 2: Бинаризация матрицы
    print("Тесты для binarize:")
    M = np.array([[1, 5], [3, 4]])
    result = binarize(MockBinarizeInput(M, 3))
    assert np.array_equal(result, np.array([[0, 1], [0, 1]])), "Бинаризация"
    assert np.array_equal(M, np.array([[1, 5], [3, 4]])), "Исходная матрица не изменена"
    print("✅ Все тесты пройдены\n")
    
    # Задача 3: Уникальные элементы строк и столбцов
    print("Тесты для unique_rows и unique_columns:")
    M = np.array([[1, 2, 2], [3, 3, 4]])
    assert unique_rows(MockMatrixInput(M)) == [[1, 2], [3, 4]], "Уникальные строки"
    assert unique_columns(MockMatrixInput(M)) == [[1, 3], [2, 3], [2, 4]], "Уникальные столбцы"
    print("✅ Все тесты пройдены\n")
    
    # Задача 5: Шахматная матрица
    print("Тесты для chess:")
    result = chess(MockChessInput(3, 4, 0, 1))
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(result, expected), "Шахматная матрица"
    print("✅ Все тесты пройдены\n")
    
    # Задача 6: Прямоугольник и эллипс
    print("Тесты для draw_rectangle и draw_ellipse:")
    rect = draw_rectangle(MockRectangleInput(4, 4, 10, 10, (255, 0, 0), (0, 0, 0)))
    assert rect.shape == (10, 10, 3), "Форма изображения прямоугольника"
    assert rect[5, 5, 0] == 255, "Центр должен быть красным"
    
    ellipse = draw_ellipse(MockEllipseInput(3, 3, 10, 10, (0, 255, 0), (0, 0, 0)))
    assert ellipse.shape == (10, 10, 3), "Форма изображения эллипса"
    assert ellipse[5, 5, 1] == 255, "Центр должен быть зелёным"
    print("✅ Все тесты пройдены\n")
    
    # Задача 7: Анализ временного ряда
    print("Тесты для analyze_time_series:")
    series = [1, 3, 2, 4, 1, 5, 2]
    result = analyze_time_series(MockTimeSeriesInput(series, 2))
    assert result.local_max_indices == [1, 3, 5], f"Локальные максимумы: {result.local_max_indices}"
    assert result.local_min_indices == [2, 4], f"Локальные минимумы: {result.local_min_indices}"
    assert len(result.moving_average) == len(series) - 2 + 1, "Длина скользящего среднего"
    print("✅ Все тесты пройдены\n")
    
    # Задача 8: One-hot encoding
    print("Тесты для one_hot:")
    labels = [0, 2, 3, 0]
    result = one_hot(MockOneHotInput(labels, None))
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert np.array_equal(result, expected), "One-hot encoding"
    print("✅ Все тесты пройдены\n")
    
    # Тест гистограмм (Задача 4)
    print("Тест для matrix_statistics и гистограмм:")
    stats = matrix_statistics(MockRandomMatrixInput(10, 10, 0, 1, 42))
    assert len(stats.row_means) == 10, "10 средних для строк"
    assert len(stats.col_means) == 10, "10 средних для столбцов"
    print("✅ Статистики вычислены корректно")
    
    # Построение гистограмм
    try:
        plot_histograms(stats)
    except Exception as e:
        print(f"⚠️ Гистограммы не построены (возможно, matplotlib не установлен): {e}")
    
    print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
