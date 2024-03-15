from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300, loss_threshold: float = 1e-10):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.loss_threshold: float = loss_threshold
        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # Рассчитываем значение функции потерь
        prev_loss = self.calc_loss(x, y)
        self.loss_history.append(prev_loss)
        
        # Переменная для хранения предыдущих значений весов
        prev_w = self.descent.w.copy()
        # Цикл по количеству итераций
        for _ in range(self.max_iter):
            # Рассчитываем градиент
            gradient = self.descent.calc_gradient(x, y)

            # Обновляем веса
            self.descent.w += self.descent.update_weights(gradient)

            # Рассчитываем значение функции потерь
            loss = self.calc_loss(x, y)

            # Добавляем значение функции потерь в историю
            self.loss_history.append(loss)

            # Проверяем условие остановки
            if np.linalg.norm(self.descent.w - prev_w) ** 2 < self.tolerance:
                break
                
            # Проверка на наличие NaN значений
            if np.isnan(np.sum(self.descent.w)):
                break          
            
            # если ошибка начинает очень сильно расти
            if loss > self.loss_threshold and loss > prev_loss:
                print("Gradient explosion detected. Stopping iterations.")
                break
                
            # Обновляем предыдущие значения весов
            prev_w = self.descent.w.copy()

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
