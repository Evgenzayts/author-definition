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
├── authorship.py              # основной файл  
├── authorship_features.py     # вспомогательный модуль  
├── your_dataset.csv           # твой датасет  
├── unknown_text.txt           # текст для предсказания (опционально)  
├── model.pkl  
├── tfidf_word.pkl  
├── tfidf_pos.pkl  
├── scaler.pkl  
