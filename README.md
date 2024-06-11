# 0. Рекомендация
Откройте jupiter notebook у себя или в гугл коллабе \
https://colab.research.google.com/drive/1fobS6dlVt4u7gteZJ1jGVdz_dDoDCFIX?usp=sharing\

**Ведь тут по большей степени выдержка материала из итогового отчёта**  

Которые вы сами можете найти тут 

[CourseworkReport.docx](CourseworkReport.docx)

[CourseworkReport.pdf](%CA%F3%F0%F1%EE%E2%EE%E9.%C3%E0%E9%ED%F3%F2%E4%E8%ED%EE%E2.%D1%E2%E5%F2%E8%EC%EE%F1%F2%FC%C0%E1%F1%EE%EB%FE%F2%ED%EE%D7%E5%F0%ED%EE%E3%EE%D2%E5%EB%E0.pdf)

# 1. Постановка задачи

Абсолютно черное тело излучает энергию пропорционально четвертой степени
температуры.

$$E = 36.9*10^{- 12}T^{4}$$

(Е -мощность излучения в вт/см, Т-температура в градусах Кельвина).

Часть общей энергии, заключенная в видимом спектре частот, с длиной
волны от $\lambda_{1}$ до $\lambda_{2}$ находится интегрированием
уравнения Планка:

$$
{E_{видимая} = E_{в} = \int_{\lambda_{1}}^{\lambda_{2}}\frac{2.39*10^{- 11}dx}{x^{5}\left( e^{\frac{1.432}{Tx}} - 1 \right)}}$$

Тогда светимость (в процентах) рассчитывается по формуле:

$${EFF = \frac{E_{в}}{E}*100\% = \frac{64.77}{T^{4}}*\int_{\lambda_{1}}^{\lambda_{2}}\frac{dx}{x^{5}\left( e^{\frac{1.432}{Tx}} - 1 \right)}}$$

Значения длин волн ($\lambda_{1}$, $\lambda_{2}$) задаются следующим
образом:

$$Значение\ \lambda_{1} = y*31.66675*10^{- 5},$$

$$\, где\ y\, - \, корень\, уравнения:2*\sqrt{x} = \cos\frac{\pi*x}{2}$$

$$
{Значение\ \lambda_{2} = - z*3.039830*10^{- 5},}$$

$$\, где\, z\, - \, значение,\, минимизирующее\, f(z)\, на\,\lbrack - 2, - 1\rbrack:$$

$$f(z) = \left( e^{z} \right)*\left( 2*z^{2} - 4 \right) + \left( 2*z^{2} - 1 \right)^{2} + e^{2*z} - 3*z^{4}$$

**Вычислить EFF** в диапазоне температур от Т=1000К до Т=9000К с шагом
1000К. По полученным данным *построить график* светимости от
температуры, взяв достаточное количество точек. *Оценить погрешность*
результата и *влияние* на точность погрешности в задании
$\lambda_{1}и\ \lambda_{2}$.

# 2. Нахождение длин волн

Для вычислений, в том случае, когда $\lambda_{2} < \lambda_{1}$ нам
придется поменять лямбды местами, чтобы лямбда первая была меньше лямбды
второй. Иначе рискуем найти отрицательную светимость.

Для решения поставленной задачи воспользуемся такими инструментами, как
Python и Wolfram Alpha. А в частности, для нахождения длин волн,
библиотекой scipy.

Импортируем необходимые модули и библиотеки, зададимся точностью
${accuracy\  = \ 10}^{- 16}$. Также напишем сами функции f(x) и f(z).

```python
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.integrate import quad
import numpy as np
from math import sqrt, cos, pi

accuracy = 1e-16
```

```python
  
def fun_x(x):  
    return 2 * sqrt(x[0]) - cos((pi * x[0]) / 2)  
  
  
def fun_z(z):  
    return (np.e ** z) * (2 * z ** 2 - 4) + (2 * z ** 2 - 1) ** 2 + np.e ** (2 * z) - 3 * z ** 4
```

Найдем х используя метод scipy.optimize.fsolve, который находит корни
уравнений func(x) = 0 Под капотом fsolve использует метод поиска с
переменной точкой, реализованный в алгоритмах hybrd и hybrj из MINPACK и
по сути является их оберткой.

Для работы fsolve также требуется начальное приближение, возьмем число 0
в виду того, что квадратный корень из x определен только на
положительном промежутке, а косинус в нуле равен единице и с увеличением
x будет уменьшаться. Получаем:

```python
res_x = optimize.fsolve(fun_x, np.array([0]))  
print(res_x)  
# [0.22105064]
```
Получаем решение: x = 0.22105064

Найдем значение z, минимизирующее f(z) на промежутке \[-2,-1\]. Для
этого воспользуемся методом scipy.optimize.minimize, а в частности
методом 'TNC' (Truncated Newton), который сочетает в себе преимущества
метода Нелдера-Мида (метод симплексного поиска) и метода сопряженных
градиентов, однако все также требует начального приближения. В данном
случае зададим границы \[-2, -1\], а начальное приближение выберем в
качестве среднего арифметического этих границ.

```python
lower_bound = -2  
upper_bound = -1  
bounds = optimize.Bounds(lower_bound, upper_bound)  
res_z = optimize.minimize(fun_z,  
                          np.array((lower_bound + upper_bound) / 2),  
                          bounds=bounds,  
                          method='TNC',  
                          tol=accuracy).x  
print(res_z)  
# [-1.31597378]
```

Получаем минимизирующее значение: z = -1.31597378

Вычислим полученные лямбды по формулам. Заметим, что для того, чтобы
наши расчеты имели какой-то смысл, $\lambda_{1}$ должна быть меньше
$\lambda_{2}$.

Поэтому отсортируем их в порядке возрастания.
```python
L_1 = res_x[0] * 31.66675 * 10 ** (-5)  
L_2 = -res_z[0] * 3.039830 * 10 ** (-5)  
L_1, L_2 = sorted([L_1, L_2])  
print(f'TEST: lambda_1 = {L_1}; lambda_2 = {L_2}')  
  
# lambda_1 = 4.000336578604378e-05  
# lambda_2 = 6.999955338251561e-05
```

В результате:

$$\lambda_{1} = 4.000336578604378*10^{- 5}$$

$$\lambda_{2} = 6.999955338251561*10^{- 5}$$

Полученные значения действительно находятся в видимом диапазоне и
представляют собой длины волн красного и фиолетового цвета
соответственно (см. рис. 2.1).

![Pasted image 20240611194541.png](AttachFolder%2FPasted%20image%2020240611194541.png)
 <div align=center>   Рис. 2.1 Таблица характеристики длин волн видимого диапазона </div>

Для повторного поиска лямбд напишем функцию, благодаря которой мы сможем
находить лямбды с разной точностью.

```python
def get_lambdas(fun_x, fun_z, accuracy=1e-16):  
    res_x = optimize.fsolve(fun_x, np.array([0]), xtol=accuracy)  
    print(f'INFO: res_x = {res_x}')  
    # [0.22105064]  
  
    lower_bound = -2  
    upper_bound = -1  
    bounds = optimize.Bounds(lower_bound, upper_bound)  
    res_z = optimize.minimize(fun_z,  
                              np.array((lower_bound + upper_bound) / 2),  
                              bounds=bounds,  
                              method='TNC',  
                              tol=accuracy).x  
    print(f'INFO: res_z = {res_z}')  
    # [-1.3159745]  
    L_1 = res_x[0] * 31.66675 * 10 ** (-5)  
    L_2 = -res_z[0] * 3.039830 * 10 ** (-5)  
    L_1, L_2 = sorted([L_1, L_2])  
    print(f'INFO: lambda_1 = {L_1}; lambda_2 = {L_2}')  
  
    # lambda_1 = 4.000336578604378e-05  
    # lambda_2 = 6.999955338251561e-05    return L_1, L_2
```


# 3. Исследование зависимости светимости от температуры и погрешности заданных длин волн

Для поиска светимости EFF требуется взять интеграл. Для этого
воспользуемся методом quad из библиотеки scipy.integrate. "quad"
использует адаптивные алгоритмы для аппроксимации определенного
интеграла функции. То есть он разбивает область интегрирования на мелкие
интервалы и вычисляет приближенное значение интеграла на каждом
интервале. Также он может принимать дополнительные аргументы для функции
интегрирования, которые можно передать с помощью параметра args, в нашем
случае переменную - T.

```python
def integrand(x, T):  
    return 1 / ((x ** 5) * (np.exp(1.432 / (T * x)) - 1))  
  
  
def EFF(T, l_bound, u_bound):  
    return (  
            64.77 / (T ** 4) * quad(integrand, l_bound, u_bound, args=T, epsabs=accuracy)[0]  
    )
```

Для проверки попробуем взять тот же интеграл с помощью WolframAlpha:

Введя: integrate from 4.000336578604378\*10\^(-5) to
6.999955338251561\*10\^(-5): dx/((x\^5) \* (e\^(1.432 / (1000 \* x)) -
1))

Получаем:

**WolframAlpha** 3085873.2325267694

**Quad** 3085873.232526763
___
```python
T = 1000.0  
lambda_1, lambda_2 = get_lambdas(fun_x, fun_z, accuracy)  
print(  
    f'TEST:\n!wa integrate from {lambda_1} to {lambda_2} dx / ((x^5)(e^((1.432)/{T} * x) - 1) = {quad(integrand, lambda_1, lambda_2, args=T, epsabs=accuracy)[0]}')  
# Получаем такой запрос для WolframAlpha  
# integrate from 4.000336578604378*10^(-5)  to  6.999955338251561*10^(-5): dx/((x^5) * (e^(1.432 / (1000 * x)) - 1))  
# 3085873.2325267694 - вычисленный интеграл в WolframAlpha  
# Или, как он выводит - 3.08587×10^6  
  
# Значения должны совпадать (при тестировании было достигнуто совпадение до 3-х и более знаков после запятой)

```
___

Было достигнуто совпадение до 7 цифры после запятой, поэтому продолжим
использовать "quad" для всего диапазона T.

```python
def calculate_EFF(lambda_1, lambda_2, label='EFF'):  
    ans = []  
    T0 = 1000  
    step = 1000  
    for i in range(9):  
        T = T0 + i * step  
        # t_res = EFF(T, lambda_1, lambda_2)  
        ans.append(EFF(T, lambda_1, lambda_2))  
  
    T_values = list(range(T0, T0 + step * len(ans), step))  
  
    return np.array(ans), T_values
    
```
Вычислим светимость EFF с исходными значениями $\mathbf{\lambda}$

```python
lambda_1, lambda_2 = get_lambdas(fun_x, fun_z, accuracy)  
# Вычислим EFF с исходными значениями lambda  
original_EFF, T_values = calculate_EFF(lambda_1, lambda_2)
```

![Pasted image 20240611195826.png](AttachFolder%2FPasted%20image%2020240611195826.png)![[Pasted image 20240611195826.png|centered|300x100]]
<div align=center> Рис. 3.1 Результаты вычисления светимости </div>

Теперь зададимся некоторым случайным отклонением от исходных значений
$\mathbf{\lambda}$

delta = 1e-6 *\# Вместо дельты может быть погрешностью измерений прибора
или ошибка округления, поэтому мы также добавим случайность в
уравнение.
```python
delta = 1e-6  # Вместо дельты может быть погрешностью измерений прибора или ошибка округления, поэтому мы также добавим случайность в уравнение.  
# delta default: 1e-6  
accuracy = 1e-6  
#%%  
# Получаем  
new_lambdas = get_lambdas(fun_x, fun_z, accuracy)  
lambda_1_new = new_lambdas[0] + delta * np.random.uniform(-1, 1)  # lambda_1 * 0.01  
lambda_2_new = new_lambdas[1] + delta * np.random.uniform(-1, 1)  # lambda_2 * 0.01  
  
print(f"INFO: Original lambda_1: {lambda_1}, Original lambda_2: {lambda_2}")  
print(f"INFO: New lambda_1: {lambda_1_new}, New lambda_2: {lambda_2_new}")  
print(f"INFO: ABS Difference in lambda_1: {np.abs(lambda_1 - lambda_1_new)}")  
print(f"INFO: ABS Difference in lambda_2: {np.abs(lambda_2 - lambda_2_new)}")
```

![Pasted image 20240611200331.png](AttachFolder%2FPasted%20image%2020240611200331.png)

Рис. 3.2 Значение новых $\mathbf{\lambda}$ и их абсолютная разница

Вычислим EFF для новых значений $\mathbf{\lambda}$, а также абсолютную
разность между оригинальным значением и новым значением EFF:

```python
new_EFF, T_values2 = calculate_EFF(lambda_1_new, lambda_2_new, label='New EFF')
diff_EFF = np.abs(original_EFF - new_EFF)  
div_diff_EFF = diff_EFF / original_EFF * 100
```

![Pasted image 20240611200318.png](AttachFolder%2FPasted%20image%2020240611200318.png)

Рис 3.3 Значения исходной светимости EFF, новой светимости для
измененных $\mathbf{\lambda}$, их абсолютная разница и относительная
абсолютная разница.

Теперь, когда нам известны все необходимые данные, мы можем нарисовать
их графики. Для этого воспользуемся библиотекой matplotlib и модулем
pyplot. Результаты работы программы при вышеописанных данных
проиллюстрированы на рисунках 3.4, 3.5, 3.6. Однако они могут меняться
от запуска к запуску, ввиду случайности значения новых
$\mathbf{\lambda}$.

По рис. 3.4 видно, что максимум светимости достигается при температуре
T=7000K и равен 39.3%. Для светимости с введенной погрешностью значение
максимума тоже достигается при данной температуре. Это объясняет, почему
дневной свет, имеющий близкую цветовую температуру (рис.1.2), кажется
нам наиболее ярким.

![Pasted image 20240611200310.png](AttachFolder%2FPasted%20image%2020240611200310.png)

Рис. 3.4 Сравнительный график значений оригинальной светимости
(*Original EFF*) и новой светимости (*New EFF*) в зависимости от
температуры

При значении delta=$10^{- 6}$ и accuracy=$10^{- 16}$ тенденция
абсолютная погрешности результата такова, что она невелика при малых
температурах, однако с их увеличением она соответственно возрастает.

![Pasted image 20240611200253.png](AttachFolder%2FPasted%20image%2020240611200253.png)

Рис. 3.4 График абсолютной погрешности в зависимости от температуры

Изменение в погрешности вычисления лямбд, при максимальной разности
погрешности в $10^{- 6}$ даёт высокую относительную абсолютную
погрешность для малых температур -- 12.55%, когда значение оригинальной
светимости слишком мало. Однако, оно постепенно уменьшается до 1--2% при
более высоких температурах.

![Pasted image 20240611200246.png](AttachFolder%2FPasted%20image%2020240611200246.png)

Рис. 3.4 Относительная абсолютная погрешность в зависимости от
температуры


# ПРИЛОЖЕНИЕ

Jupyter notebook:

<https://colab.research.google.com/drive/1fobS6dlVt4u7gteZJ1jGVdz_dDoDCFIX?usp=sharing>

Листинг кода


Код решающий задачу о нахождении светимости (в процентах) при заданной
температуре и заданных\
lambda 1 и lambda 2\
Код хорошо задокументирован в самой курсовой работе, а также в
соответствующим юпитер блокноте\
\
https://colab.research.google.com/drive/1fobS6dlVt4u7gteZJ1jGVdz_dDoDCFIX?usp=sharing\


```python
"""  
Код решающий задачу о нахождении светимости (в процентах) при заданной температуре и заданных  
lambda 1 и lambda 2  
Код хорошо задокументирован в самой курсовой работе, а также в соответствующим юпитер блокноте  
  
https://colab.research.google.com/drive/1fobS6dlVt4u7gteZJ1jGVdz_dDoDCFIX?usp=sharing  
  
"""  
import matplotlib.pyplot as plt  
import scipy.optimize as optimize  
from scipy.integrate import quad  
import numpy as np  
from math import sqrt, cos, pi  
  
accuracy = 1e-16  
  
  
def fun_x(x):  
    return 2 * sqrt(x[0]) - cos((pi * x[0]) / 2)  
  
  
def fun_z(z):  
    return (np.e ** z) * (2 * z ** 2 - 4) + (2 * z ** 2 - 1) ** 2 + np.e ** (2 * z) - 3 * z ** 4  
  
  
def get_lambdas(fun_x, fun_z, accuracy=1e-16):  
    res_x = optimize.fsolve(fun_x, np.array([0]), xtol=accuracy)  
    print(f'INFO: res_x = {res_x}')  
    # [0.22105064]  
  
    lower_bound = -2  
    upper_bound = -1  
    bounds = optimize.Bounds(lower_bound, upper_bound)  
    res_z = optimize.minimize(fun_z,  
                              np.array((lower_bound + upper_bound) / 2),  
                              bounds=bounds,  
                              method='TNC',  
                              tol=accuracy).x  
    print(f'INFO: res_z = {res_z}')  
    # [-1.3159745]  
    L_1 = res_x[0] * 31.66675 * 10 ** (-5)  
    L_2 = -res_z[0] * 3.039830 * 10 ** (-5)  
    L_1, L_2 = sorted([L_1, L_2])  
    print(f'INFO: lambda_1 = {L_1}; lambda_2 = {L_2}')  
  
    # lambda_1 = 4.00033875088364e-05  
    # lambda_2 = 6.999955338251561e-05    return L_1, L_2  
  
  
def integrand(x, T):  
    return 1 / ((x ** 5) * (np.exp(1.432 / (T * x)) - 1))  
  
  
def EFF(T, l_bound, u_bound):  
    return (  
            64.77 / (T ** 4) * quad(integrand, l_bound, u_bound, args=T, epsabs=accuracy)[0]  
    )  
  
def calculate_EFF(lambda_1, lambda_2, label='EFF'):  
    ans = []  
    T0 = 1000  
    step = 1000  
    for i in range(9):  
        T = T0 + i * step  
        # t_res = EFF(T, lambda_1, lambda_2)  
        ans.append(EFF(T, lambda_1, lambda_2))  
  
    T_values = list(range(T0, T0 + step * len(ans), step))  
  
    return np.array(ans), T_values  
  
  
lambda_1, lambda_2 = get_lambdas(fun_x, fun_z, accuracy)  
# Вычислим EFF с исходными значениями lambda  
original_EFF, T_values = calculate_EFF(lambda_1, lambda_2)  
  
# Создадим некоторое отклонение от исходных значений lambda  
delta = 1e-6  # Вместо дельты может быть погрешностью измерений прибора или ошибка округления, поэтому мы также  
# добавим случайность в уравнение.  
# delta default: 1e-6  
accuracy = 1e-16  
# Получаем  
new_lambdas = get_lambdas(fun_x, fun_z, accuracy)  
lambda_1_new = new_lambdas[0] + delta * np.random.uniform(-1, 1)  # lambda_1 * 0.01  
lambda_2_new = new_lambdas[1] + delta * np.random.uniform(-1, 1)  # lambda_2 * 0.01  
  
print(f"INFO: Original lambda_1: {lambda_1}, Original lambda_2: {lambda_2}")  
print(f"INFO: New lambda_1: {lambda_1_new}, New lambda_2: {lambda_2_new}")  
print(f"INFO: Difference in lambda_1: {np.abs(lambda_1 - lambda_1_new)}")  
print(f"INFO: Difference in lambda_2: {np.abs(lambda_2 - lambda_2_new)}")  
# Вычислим EFF для новых значений лямбда  
new_EFF, T_values2 = calculate_EFF(lambda_1_new, lambda_2_new, label='New EFF')  
  
# Вычисляем разность между оригинальным значением and новым значением EFF  
diff_EFF = np.abs(original_EFF - new_EFF)  
div_diff_EFF = diff_EFF / original_EFF * 100  
  
# График EFF и new EFF  
plt.figure(figsize=(10, 6))  
plt.plot(T_values, original_EFF, marker='1')  
plt.title('EFF and new EFF vs T')  
plt.xlabel('Temperature' + r' ($Kelvins$)')  
plt.ylabel('EFF (%)')  
plt.plot(T_values2, new_EFF, marker='x', color='green')  
plt.legend(['Original EFF', 'New EFF'])  
plt.grid(True, alpha=0.2, linestyle=':')  
plt.show()  
  
print(f"INFO: Original EFF: {original_EFF}")  
print(f"INFO: New EFF: {new_EFF}")  
print(f"INFO: Difference in EFF: {diff_EFF}")  
print(f"INFO: div_diff_EFF: {div_diff_EFF}")  
  
# График модуля отклонения  
plt.figure(figsize=(10, 6))  
plt.plot(T_values, diff_EFF, marker="1", color='red')  
plt.title('Difference in EFF vs T')  
plt.xlabel('Temperature' + r' ($Kelvins$)')  
plt.ylabel('Difference in EFF')  
plt.grid(True, alpha=0.2, linestyle=':')  
plt.show()  
  
# Относительная погрешность  
plt.figure(figsize=(10, 6))  
plt.plot(T_values, div_diff_EFF, marker='.', color='magenta')  
plt.title('Difference in EFF (relatively) vs T')  
plt.xlabel('Temperature' + r' ($Kelvins$)')  
plt.ylabel('Difference in EFF (%)')  
plt.grid(True, alpha=0.2, linestyle=':')  
plt.show()
```