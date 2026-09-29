import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QGridLayout, QSpinBox, QLabel, QGroupBox, QLineEdit,
    QMessageBox, QDialogButtonBox, QWidget
)
from PyQt5.QtCore import Qt


class FilterCell(QLineEdit):
    """
    Одна клетка матрицы пользовательского фильтра.
    Позволяет вводить числа (включая отрицательные и дробные).
    """

    def __init__(self, row: int, col: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col

        # Настройка внешнего вида
        self.setFixedSize(60, 30)
        self.setAlignment(Qt.AlignCenter)
        self.setText("0")
        self.setStyleSheet(
            "QLineEdit { border: 1px solid #ccc; padding: 2px; }"
        )

    def get_value(self) -> float:
        """Возвращает значение клетки как float."""
        text = self.text().strip()
        if not text:
            return 0.0
        try:
            return float(text.replace(',', '.'))
        except ValueError:
            return 0.0

    def set_value(self, value: float):
        """Устанавливает значение клетки."""
        # Форматируем: если целое число — без десятичной части
        if value == int(value):
            self.setText(str(int(value)))
        else:
            self.setText(f"{value:.2f}")


class CustomFilterDialog(QDialog):
    """
    Диалог для ввода пользовательской матрицы фильтра.
    Позволяет задать размер матрицы и ввести значения в каждую клетку.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Пользовательский фильтр")
        self.matrix_size = 3
        self.cells = []

        self._setup_ui()
        self._build_grid()

    def _setup_ui(self):
        """Настраивает интерфейс диалога."""
        layout = QVBoxLayout(self)

        # Группа "Размер матрицы"
        size_group = QGroupBox("Размер матрицы")
        size_layout = QHBoxLayout(size_group)

        size_layout.addWidget(QLabel("Сторона квадрата:"))

        self.spin_size = QSpinBox()
        self.spin_size.setRange(2, 10)
        self.spin_size.setValue(3)
        self.spin_size.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.spin_size)

        layout.addWidget(size_group)

        # Группа "Матрица фильтра"
        matrix_group = QGroupBox("Матрица фильтра (введите значения)")
        matrix_layout = QVBoxLayout(matrix_group)

        # Контейнер для сетки
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(2)
        self.grid_layout.setAlignment(Qt.AlignCenter)

        matrix_layout.addWidget(self.grid_container)

        # Подсказка
        hint_label = QLabel(
            "Примеры:\n"
            "• Повышение резкости:\n"
            "  0 -1 0 / -1 5 -1 / 0 -1 0\n"
            "• Размытие (усреднение):\n"
            "  1 1 1 / 1 1 1 / 1 1 1 (затем разделить на 9)"
        )
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        matrix_layout.addWidget(hint_label)

        layout.addWidget(matrix_group)

        # Группа "Быстрые шаблоны"
        templates_group = QGroupBox("Быстрые шаблоны")
        templates_layout = QHBoxLayout(templates_group)

        btn_sharpen = QPushButton("Резкость")
        btn_sharpen.clicked.connect(self._load_sharpen)
        templates_layout.addWidget(btn_sharpen)

        btn_blur = QPushButton("Размытие")
        btn_blur.clicked.connect(self._load_blur)
        templates_layout.addWidget(btn_blur)

        btn_edge = QPushButton("Границы")
        btn_edge.clicked.connect(self._load_edge)
        templates_layout.addWidget(btn_edge)

        btn_emboss = QPushButton("Тиснение")
        btn_emboss.clicked.connect(self._load_emboss)
        templates_layout.addWidget(btn_emboss)

        layout.addWidget(templates_group)

        # Кнопки OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _build_grid(self):
        """Создаёт сетку клеток размером matrix_size × matrix_size."""
        # Очищаем старую сетку
        for row_cells in self.cells:
            for cell in row_cells:
                self.grid_layout.removeWidget(cell)
                cell.setParent(None)
                cell.deleteLater()
        self.cells.clear()

        # Создаём новые клетки
        for row in range(self.matrix_size):
            row_cells = []
            for col in range(self.matrix_size):
                cell = FilterCell(row, col)
                self.grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self.cells.append(row_cells)

    def _on_size_changed(self, new_size: int):
        """Обработчик изменения размера матрицы."""
        self.matrix_size = new_size
        self._build_grid()

    def _load_sharpen(self):
        """Загружает шаблон повышения резкости."""
        if self.matrix_size != 3:
            QMessageBox.warning(self, "Внимание", "Шаблон работает только для матрицы 3×3")
            return

        template = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
        self._load_template(template)

    def _load_blur(self):
        """Загружает шаблон размытия (усреднение)."""
        if self.matrix_size != 3:
            QMessageBox.warning(self, "Внимание", "Шаблон работает только для матрицы 3×3")
            return

        # Для усреднения все элементы = 1/9, но для простоты вводим 1
        # (нормализацию можно сделать автоматически)
        template = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        self._load_template(template)

    def _load_edge(self):
        """Загружает шаблон детекции границ (Лапласиан)."""
        if self.matrix_size != 3:
            QMessageBox.warning(self, "Внимание", "Шаблон работает только для матрицы 3×3")
            return

        template = [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
        self._load_template(template)

    def _load_emboss(self):
        """Загружает шаблон тиснения."""
        if self.matrix_size != 3:
            QMessageBox.warning(self, "Внимание", "Шаблон работает только для матрицы 3×3")
            return

        template = [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ]
        self._load_template(template)

    def _load_template(self, template: list):
        """Загружает шаблон в клетки матрицы."""
        for row_idx, row_data in enumerate(template):
            for col_idx, value in enumerate(row_data):
                self.cells[row_idx][col_idx].set_value(value)

    def get_kernel(self) -> np.ndarray:
        """
        Возвращает введённую матрицу как numpy-массив.
        Тип данных: float32 (поддерживает отрицательные и дробные значения).
        """
        kernel = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32)
        for row_idx, row_cells in enumerate(self.cells):
            for col_idx, cell in enumerate(row_cells):
                kernel[row_idx, col_idx] = cell.get_value()
        return kernel