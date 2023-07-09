## Сервис для запуска моделей торговых новостей
Для запуска сервиса необходимо задать переменные окружения в .env файле
```python
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=
POSTGRES_PORT=
PATH_NER=/app/model0
PATH_ER=/app/model1
PATH_CLF_NEWS=/app/model2
PATH_WORD_PROD_CLF=/app/model3
PATH_TEXT_SPEC_PROD_CLF=/app/model4
PATH_LANG=/app/model5
S3_ACCESS_KEY=
S3_SECRET_KEY=
```
Для доступа к моделям необходимо иметь логин и пароль к MINIO, из которого модели скачиваются при сборке контейнера:</br> http://10.8.0.2:9000/minio/models/ 
</br>
S3_ACCESS_KEY - логин для доступа к MINIO; </br>
S3_SECRET_KEY - пароль для доступа к MINIO;

Пути к моделям, загружаемых для инференса, прописываются в файле s3_models.txt в следующем порядке:
```python
0. Модель распознавания сущностей (PATH_NER)
1. Модель классификации отношений между сущностями (PATH_ER)
2. Модель классификации новостей (PATH_CLF_NEWS)
3. Модель классификации товаров по СМТК по названию товара (PATH_WORD_PROD_CLF)
4. Модель классификации текста по группе СМТК 
"93 - Специальные операции и товары, не классифицированные по типу"
(бинарная классификация по тексту новости) (PATH_TEXT_SPEC_PROD_CLF)
5. Модель определения языка текста (PATH_LANG)

```
Переменные доступа к базе данных нужны для семантического поиска дубликатов новостей в базе.

После запуска доступны следующие маршруты API:
```python
/infer/ner - распознавания сущностей
/infer/relation_extraction - Модель классификации отношений между сущностями
/infer/clf_news - Модель классификации новостей
/infer/word_clf - Модель классификации товаров по СМТК по названию товара
/infer/text_clf - Модель классификации текста по группе СМТК
/infer/lang_clf - Модель определения языка текста
/get_similiar - семантический поиск дубликатов
/embedding - получение вектора текста
```


###NER


Сервер: `iori2`

Адрес: `http://10.8.0.5:8989/infer/ner`

Пример запроса:

```
sentence = "« Сила Сибири » — газотранспортная система , предполагающая транспортировку газа Якутского ( на базе Чаяндинского ме
   ...: сторождения , запасы газа — 1,2 триллиона кубометров ) и Иркутского ( на базе Ковыктинского месторождения , запасы газа — 1,5 
   ...: триллиона кубометров ) центров газодобычи на Дальний Восток России и в Китай"
response = requests.post(NER_SERVER, json={"text": sentence})
```

Пример ответа:
```
[{'label': 'SELLS_TO',
  'span1': [0, 0],
  'span2': [2, 2],
  'ner1': 'COUNTRY',
  'ner2': 'PRODUCT',
  'entity1': 'Россия',
  'entity2': 'танки',
  'logits': {'no_relation': 0.09,
   'WORKS_AS': 0.03,
   'WORKPLACE': 0.0,
   'OWNERSHIP': 0.06999999999999999,
   'HEADQUARTERED_IN': 0.27,
   'SELLS_TO': 98.42,
   'EVENT_TAKES_PART_IN': 0.04,
   'ALTERNATIVE_NAME': 0.06999999999999999,
   'MEMBER': 0.03,
   'ABBREVIATION': 0.02,
   'TAKES_PLACE_IN': 0.03,
   'ORGANIZES': 0.01,
   'PRODUCES': 0.31,
   'ORIGINS_FROM': 0.02,
   'DATE_TAKES_PLACE_ON': 0.01,
   'SUBORDINATE_OF': 0.01,
   'PARENT_OF': 0.03,
   'FOUNDED_BY': 0.01,
   'DATE_FOUNDED_IN': 0.02,
   'NUMBER_OF_EMPLOYEES_FIRED': 0.01,
   'SUBEVENT_OF': 0.01,
   'PLACE_RESIDES_IN': 0.05,
   'ACQUINTANCE_OF': 0.05,
   'NUMBER_OF_EMPLOYEES': 0.02,
   'DATE_DEFUNCT_IN': 0.02,
   'AGE_IS': 0.04,
   'REFERENCE': 0.04,
   'REFERENCES': 0.04,
   'DATE_OF_BIRTH': 0.02,
   'SELLS': 0.05,
   'SIBLING': 0.04,
   'BUYS': 0.06999999999999999,
   'RELATIVE': 0.03,
   'BORN_IN': 0.03}}]
```
###Relation extraction



Сервер: `iori2`

Адрес: `http://10.8.0.5:8989/infer/relation_extraction`


Пример запроса:
```
REL_SERVER = "http://10.8.0.5:8989/infer/relation_extraction"
requests.post(REL_SERVER, json={"example": {"token": ["Россия", "продает", "танки", "Германии"], "subj_start": 0, "subj_end":
   ...: 0, "obj_start":2, "obj_end":2, "subj_type":"COUNTRY", "obj_type":"PRODUCT"}}).json()
```

Пример ответа:
```
[{'entity': 'FAC',
  'score': 0.4006073673566182,
  'index': 2,
  'word': 'Сила Сибири',
  'start': 2,
  'end': 13},
 {'entity': 'CITY',
  'score': 0.4238370756308238,
  'index': 22,
  'word': 'Якутского',
  'start': 81,
  'end': 90},
 {'entity': 'LOCATION',
  'score': 0.5029935936133066,
  'index': 28,
  'word': 'Чаяндинского месторождения',
  'start': 101,
  'end': 127},
 {'entity': 'QUANTITY',
  'score': 0.9006648540496827,
  'index': 39,
  'word': '1,2 триллиона кубометров',
  'start': 144,
  'end': 168},
 {'entity': 'CITY',
  'score': 0.5304505527019501,
  'index': 51,
  'word': 'Иркутского',
  'start': 173,
  'end': 183},
 {'entity': 'LOCATION',
  'score': 0.5355854153633117,
  'index': 57,
  'word': 'Ковыктинского место',
  'start': 194,
  'end': 213},
 {'entity': 'QUANTITY',
  'score': 0.8990582406520844,
  'index': 68,
  'word': '1,5 триллиона кубометров',
  'start': 238,
  'end': 262},
 {'entity': 'REGION',
  'score': 0.33744708200295764,
  'index': 86,
  'word': 'Дальний Восток',
  'start': 287,
  'end': 301},
 {'entity': 'COUNTRY',
  'score': 0.9924811124801636,
  'index': 89,
  'word': 'России',
  'start': 302,
  'end': 308},
 {'entity': 'COUNTRY',
  'score': 0.9969527125358582,
  'index': 92,
  'word': 'Китай',
  'start': 313,
  'end': 318}]
```
## Результаты 

Модели: https://ranepa-my.sharepoint.com/:f:/g/personal/pletenev-sa_ranepa_ru/EosbCuMGUFhFuTkyszygmeUBzjQexCzNkt8hY-zIFzReiw?e=skREL3

|model                  |batch_size|sequence_length|MEMORY|TIME  |ner        |rured ner  |
|-----------------------|----------|---------------|------|------|-----------|-----------|
|ro_bert_088_ner_rured  |1         |8              |1902  |0.1908|0.88       |0.916687591|
|ro_bert_088_ner_rured  |1         |32             |1902  |0.3215|           |           |
|ro_bert_088_ner_rured  |1         |128            |1902  |0.7124|           |           |
|rured2021_01_12        |1         |8              |1024  |0.0475|0.843903659|           |
|rured2021_01_12        |1         |32             |1029  |0.0621|           |           |
|rured2021_01_12        |1         |128            |1034  |0.1469|           |           |
|tiny_bert_083_ner_rured|1         |8              |428   |0.0069|0.828      |0.869117813|
|tiny_bert_083_ner_rured|1         |32             |428   |0.0095|           |           |
|tiny_bert_083_ner_rured|1         |128            |430   |0.0231|           |           |