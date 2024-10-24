# Классификация временных рядов

## Структура проекта

- **main.py**: Скрипт для преобразования данных и вывода предсказаний в файл `submission.csv`.
- **analysis.ipynb**: Jupyter Notebook с EDA анализом и проверкой гипотез.
- **data_preprocessing.py**: Скрипт для предобработки данных.
- **model.py**: Содержит архитектуру нейронной сети.
- **requirements.txt**: Список необходимых библиотек.
- **trained_model**: Файл с обученными весами модели.

## Метрики на валидационной выборке
- **ROC-AUC**: `0.89`
- **Binary Cross-Entropy Loss**: `0.35`

## Инструкция по установке

1. **Клонирование репозитория:**

   Склонируйте проект с GitHub:
   ```
   git clone https://github.com/kuzminov-al/vk_test.git
   cd vk_test
   ```


2. **Создание виртуального окружения:**
    ```
    python -m venv venv
    ```

    После этого активируйте виртуальное окружение с помощью команды:

    ```
    source venv/bin/activate  # для Linux/Mac
    venv\Scripts\activate     # для Windows
    ```


3. **Установка зависимостей:**
    ```
    pip install -r requirements.txt
    ```


4. **Запуск проекта:**
    Для запуска основного скрипта и получения предсказаний выполните:
    ```
    python main.py
    ```
    Модель автоматически загрузит обученные веса из файла trained_model, предобработает данные и выведет предсказания в файл submission.csv.
