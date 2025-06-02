# author-definition

**Обучение модели на стихах**
python3 authorship.py --train

**Предсказание автора для нового текста**
python3 authorship.py --predict path/to/unknown_poem.txt

### Структура проекта

.  
├── authorship.py                       # основной файл с обучением и предсказанием  
├── data                                # датасет с текстами и авторами  
    ├── автор  
        ├── стих  
        ├── стих  
        ├── ...  
    ├── автор  
    ├── ...  
├── unknown_text                        # текст для предсказания (опционально)  
├── model.pkl                           # обученная модель (Logistic Regression)  
├── tfidf_word.pkl                      # TF-IDF-векторизатор по словам  
├── tfidf_pos.pkl                       # TF-IDF-векторизатор по POS-тегам  
├── scaler.pkl                          # стандартизатор признаков  

