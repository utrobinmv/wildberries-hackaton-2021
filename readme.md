**Запуск сервера**

```
pip install -r requirements.txt
python server.py
```



**Перед запуском**

С сервера скачайте дополнительные файлы по ссылке:

https://disk.yandex.ru/d/5ogVcMTDEAqvhA

И приложите датасеты от Хакатона в папку dataset





**Поэтапная подготовительная воспроизведения работа алгоритма на других данных**

Шаг 1 - Создание классификатора ОКПД + КТРУ

```
01_read_csv.ipynb
```

Шаг 2 - Векторизация таблицы query_popularity

```
03_read_csv-dataset-01.ipynb
```

Шаг 3 - Векторизация таблицы search_history

```
03_read_csv-dataset-02.ipynb
```

Шаг 4 - Объединение таблицы ОКПД + КТРУ + query_popularity

```
04-compare-quary.ipynb
```

Шаг 5 - Объединение таблицы ОКПД + КТРУ + search_history

```
04-compare-search_history.ipynb
```

Шаг 6 - Тестирование выдачи с сайта

Запуск сервера
