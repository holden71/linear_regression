### Імпорт бібліотек та функцій
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
###


### Функція побудови та тренування моделі
def train_model(data):
    y = data["y"]
    X = data.iloc[:, 0:1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=69)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    determ_coef = regression_model.score(X_train, y_train)

    return regression_model, determ_coef
###


# y_predict = regression_model.predict(X_test)
### Прогнозування навчальних величин
# test_x = test_dataset['x']
# test_y = test_dataset['y']
#
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_test, y_predict)
# plt.xlabel('X')
# plt.ylabel('Y')
#
# plt.title('Прогнозування на навчальних даних', fontsize=12)
#
# plt.show()
###


### Прогнозування тестових величин
# test_x = test_dataset['x']
# test_y = test_dataset['y']
#
# plt.scatter(test_x, test_y, color='black')
# plt.plot(X_test, y_predict)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Прогнозування на тестових даних', fontsize=12)
#
# plt.show()
###

### Допоміжні функції
def choose_correct_int(title = '', min_value = -10000):
    while True:
        try:
            new_value = int(input(title))

            if new_value < min_value:
                print('Мала кількість зразків!')
            else:
                return new_value
        except:
            print('Некоректне число!')

def choose_correct_float(title=''):
    while True:
        try:
            return float(input(title))
        except:
            print('Некоректне число!')

def linear_function(linear_model, argument):
    return linear_model.intercept_ + linear_model.coef_[0] * argument
###


### Меню взаємодії з користувачем
def main_menu(data=pd.DataFrame()):
    print('\n' * 50)
    print('Даниленко Кирилл КМ-92')
    print('Прогнозовуння економічних показників підприємства')
    print('\nДоступні дії:')

    if data.empty:
        menu_items = [
            '1. Ввести навчальні дані',
        ]
    else:
        menu_items = [
            '1. Змінити навчальні дані',
            '2. Відобразити навчальні дані',
            '3. Очистити навчальні дані',
            '4. Перевірити кореляцію даних',
            '5. Навчити модель прогнозування',
            '6. Спрогнозувати економічні показники',
            '7. Побудувати графік регресійного рівняння',
        ]

    for item in menu_items:
        print(item)
    user_choice = choose_correct_int('\nОберіть один з пунктів: ')

    if user_choice == 1:
        print('\n' * 50)
        menu_items = [
            '1. Ручне введення',
            '2. Імпорт з файлу',
            '3. Повернутися до головного меню'
        ]
        for item in menu_items:
            print(item)
        user_choice = choose_correct_int('\nОберіть один з пунктів: ')

        if user_choice == 1:
            print('\n' * 50)
            row_amount = choose_correct_int('\nВведіть кількість зразків (Мін. кількість 4): ', 4)

            xs, ys = [], []
            for i in range(row_amount):
                print('\n')
                print(f'Зразок {i + 1}/{row_amount}')
                xs.append(choose_correct_float(f'Введіть X{i}: '))
                ys.append(choose_correct_float(f'Введіть Y{i}: '))

            data = pd.DataFrame(list(zip(xs, ys)), columns=['x', 'y'])
            data = data.sort_values(by=['x'])

        elif user_choice == 2:
            print('\n' * 50)
            while True:
                try:
                    data = pd.read_csv(input('Введіть повну назву файлу (Напр. train.csv): '))
                    data = data.sort_values(by=['x'])
                    break
                except:
                    print('Такого файлу не існує!\n')
        elif user_choice == 4:
            data = pd.read_csv('train.csv')
            data = data.sort_values(by=['x'])
        else:
            main_menu(data)
    elif user_choice == 2 and not data.empty:
        print('\n' * 50)
        print(f'Навчальні дані:\n{data}')
    elif user_choice == 3 and not data.empty:
        print('\n' * 50)

        temp = np.sum(np.isnan(data)).y

        print(f'Кількість записів з помилковими даними: {temp}')
        if temp != 0:
            print(f'Список помилкових записів:\n{data[data.isna().any(axis=1)]}')

            data = data.dropna()

            print('\nВказані записи успішно видалено.\n')
    elif user_choice == 4 and not data.empty:
        print('\n' * 50)

        plt.figure(figsize=(6, 4))
        X = data['x']
        Y = data['y']

        plt.scatter(X, Y)
        plt.xlabel('X (незалежні)', fontsize=10)
        plt.ylabel('Y (залежні)', fontsize=10)
        plt.title('Перевірка кореляції початкових даних', fontsize=12)
        plt.show()

        _, determ_coef = train_model(data.dropna())
        print(f'Коефіцієнт детермінації: {determ_coef}')
        if determ_coef > 0.7:
            print('Побудова регресійної моделі має сенс (коеф. > 0.7)')
        else:
            print('Побудова регресійної моделі не має сенсу (коеф. < 0.7)')

    elif user_choice == 5 and not data.empty:
        print('\n' * 50)

        data = data.dropna()
        model, determ_coef = train_model(data)
        print('Рівняння регресії побудовано успішно.')
        print(f'Коефіцієнт B0: {round(model.intercept_, 6)}')
        print(f'Коефіцієнт B1: {round(model.coef_[0], 6)}')


    elif user_choice == 6 and not data.empty:
        print('\n' * 50)

        model, determ_coef = train_model(data.dropna())

        independent_factor = choose_correct_float('Введіть значення незалежного параметру: ')
        print(f'Спрогнозоване значення: {model.coef_[0]*independent_factor + model.intercept_}')

    elif user_choice == 7 and not data.empty:
        print('\n' * 50)
        data = data.dropna()
        plt.figure(figsize=(6, 4))
        X = data['x']
        Y = data['y']
        plt.scatter(X, Y)
        model, determ_coef = train_model(data)

        plt.plot(X, linear_function(model, X), color='red')

        plt.xlabel('X (незалежні)', fontsize=10)
        plt.ylabel('Y (залежні)', fontsize=10)
        plt.title('Графік регресійного рівняння', fontsize=12)
        plt.legend(['Початкові дані', 'Прогнозуюча пряма'])

        plt.show()


    else:
        main_menu(data)


    print('\nДія виконана успішно.\nПовернутися в головне меню? (Y/N)')
    if input().lower() in ['y', 'н']:
        main_menu(data)
###


### Старт програми
main_menu()
###
