# Антагонистическая игра

## Решение

Основная функция **nash_equilibrium(A)** получает на вход платежную матрицу А типа NumPy array. 

Функция **nash_equilibrium(A)** ищет значения седловых точек. Если эта точка единственная, то она и есть решение, иначе решаем задачу в смешанных стратегиях, сводим решение матричной игры к паре двойственных задач линейного программирования и решаем с помощью **linprog** из модуля **SciPy**. Функция возвращает значение игры и векторы оптимальных стратегий для первого и второго игроков.

Запуск: в терминале пишем python matrix_game_numpy.py
На вход программе подается n -- кол-во строк матрицы и сама матрица, после чего выводятся два вектора стратегий для игроков 1 и 2, а также значение игры.


С помощью библиотеки **matplotlib** языка Python, визуализируются спектры стратегий игроков. На графиках показаны вероятности принять ту или иную стратегию для игроков 1 и 2 по оси ординат. По оси абсцисс указаны номера стратегий игроков. Приведены примеры игр, в которых:

 (1) Достигается равновесие по Нэшу

 (2) Спектр стратегий не полон

 (3) Спектр стратегий полон


## Необходимое ПО:
**Jupyter Notebook**, библиотеки **SciPy**, **NumPy**, **Matplotlib** для Python

## Инструкция по запуску:
В терминале запускаем **jupyter notebook**, выбираем файл io_prac_task1.ipynb, запустите его.

## Студенты выполнившие задание
*Титушин Александр* и *Тишина Ульяна*, 311 группа

Все этапы работы были выполненны совместно.
