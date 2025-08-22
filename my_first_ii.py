import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
from tkinter import messagebox

# 1. Подготовка данных
# Пример данных
data = pd.DataFrame({
'age': [25, 30, 22, 28, 80, 30, 25],
'experience': [3, 5, 1, 2, 2, 4, 6],
'skill': [80, 60, 75, 70, 45, 67, 55],
'fitness': [90, 70, 85, 65, 45, 70, 56],
'win': [1, 1, 0, 0, 0, 1, 1]
})

# Разделение данных на признаки (X) и метки (y)

X = data[['age', 'experience', 'skill', 'fitness']]
y = data['win']
# Разделение данных на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Нормализация данных
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# 2. Создание модели

model = Sequential()

model.add(Dense(32, input_dim=4, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# 3. Компиляция модели

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Обучение модели

model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# 5. Оценка модели

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy: {accuracy * 100:.2f}%')


# Создание графического интерфейса

def predict_win():
    try:

        age = float(age_entry.get())

        experience = float(experience_entry.get())

        skill = float(skill_entry.get())

        fitness = float(fitness_entry.get())

        new_player = pd.DataFrame({

            'age': [age],

            'experience': [experience],

            'skill': [skill],

            'fitness': [fitness]

        })

        # Нормализация данных

        new_player = scaler.transform(new_player)

        # Предсказание вероятности выигрыша

        prediction = model.predict(new_player)

        probability = prediction[0][0] * 100

        messagebox.showinfo("Prediction", f"Predicted probability of winning: {probability:.2f}%")

    except ValueError:

        messagebox.showerror("Input Error", "Please enter valid numeric values.")
# Создание окна

window = tk.Tk()

window.title("Player Win Prediction")

tk.Label(window, text="Age:").grid(row=0)
tk.Label(window, text="Experience:").grid(row=1)
tk.Label(window, text="Skill:").grid(row=2)
tk.Label(window, text="Fitness:").grid(row=3)

age_entry = tk.Entry(window)
experience_entry = tk.Entry(window)
skill_entry = tk.Entry(window)
fitness_entry = tk.Entry(window)

age_entry.grid(row=0, column=1)
experience_entry.grid(row=1, column=1)
skill_entry.grid(row=2, column=1)
fitness_entry.grid(row=3, column=1)
tk.Button(window, text="Predict Win", command=predict_win).grid(row=4, column=1, pady=10)

window.mainloop()
