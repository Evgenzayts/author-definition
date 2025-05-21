# author-definition

Обучить модель
```
python authorship.py --train
```
Определить автора текста
```
python authorship.py --predict unknown_text.txt

```
### Структура проекта

.
├── authorship.py              # основной файл с обучением и предсказанием  
├── authorship_features.py     # извлечение признаков из текста  
├── your_dataset.csv           # датасет с текстами и авторами  
├── unknown_text.txt           # текст для предсказания (опционально)  
├── model.pkl                  # обученная модель (Logistic Regression)  
├── tfidf_word.pkl             # TF-IDF-векторизатор по словам  
├── tfidf_pos.pkl              # TF-IDF-векторизатор по POS-тегам  
├── scaler.pkl                 # стандартизатор признаков  

