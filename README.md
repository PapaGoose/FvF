Устанавливаем необходимые библеотеки:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import zipfile
import csv
import sys
import os
from tqdm import tqdm
from google.colab import files

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split, StratifiedKFold

import PIL
from PIL import ImageOps, ImageFilter
#увеличим дефолтный размер графиков
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
#графики в svg выглядят более четкими
%config InlineBackend.figure_format = 'svg' 
%matplotlib inline

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Tensorflow   :', tf.__version__)
print('Keras        :', tf.keras.__version__)
```
Так как работа выполнялась в Collab, то удобнее скачивать данные прямо в ВМ:
```python
%cd ~ #Переходим в корневую директорию
```
```python
files.upload() #Загружаем заранее скачанный kaggle.json для скачивания датасетов прямо с сайта
```
```python
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c 'sf-dl-car-classification' #Скачиваем в ВМ датасеты
```
Разархивируем:
```python
! mkdir train
! unzip train.zip -d train

! mkdir test
! unzip test.zip -d test
```
Можно приступать к работе.
Выставляем начальные настройки:
```python
# В setup выносим основные настройки: так удобнее их перебирать в дальнейшем.
RANDOM_SEED = 1
EPOCHS               = 15  # эпох на обучение
BATCH_SIZE           = 32 # уменьшаем batch если сеть большая, иначе не поместится в память на GPU
LR                   = 1e-4
VAL_SPLIT            = 0.15 # сколько данных выделяем на тест = 15%

CLASS_NUM            = 10  # количество классов в нашей задаче
IMG_SIZE             = 299 # какого размера подаем изображения в сеть
IMG_CHANNELS         = 3   # у RGB 3 канала
input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
```
Загружаем данные в датафремы:
```python
train_df = pd.read_csv("train.csv")
sample_submission = pd.read_csv("sample-submission.csv")
```
Задаем данные для аугментации:
```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=VAL_SPLIT, # set validation split
    horizontal_flip=False)
```
Создаем генераторы:
```python
train_generator = train_datagen.flow_from_directory(
    'train/train',      # директория где расположены папки с картинками 
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, seed=RANDOM_SEED) # set as training data

test_generator = test_datagen.flow_from_directory(
    'train/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True, seed=RANDOM_SEED,
    subset='validation') # set as validation data

test_sub_generator = test_datagen.flow_from_dataframe( 
    dataframe=sample_submission,
    directory='test/test_upload/',
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)
```
Устанавливаем чекпоинты:
```python
checkpoint = ModelCheckpoint('best_model.hdf5' , monitor = ['val_accuracy'] , verbose = 1  , mode = 'max')
LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)#Уменьшает LR если модель метрика не улучшается
callbacks_list = [checkpoint,LR]
```
Скачиваем базовую модель и для файнтюнинга не обучаем ее слои:
```python
base_model = Xception(weights='imagenet', include_top=False, input_shape = input_shape)
base_model.trainable = False
```
Добавляем к модели новую голову и обучаем на всей тренировочной выборке:
```python
model=M.Sequential()
model.add(base_model)

model.add(L.GlobalAveragePooling2D())
model.add(L.BatchNormalization())

model.add(L.Dense(256, activation='elu'))
model.add(L.BatchNormalization())
model.add(L.Dropout(0.25))

model.add(L.Dense(64, activation='elu'))
model.add(L.BatchNormalization())
model.add(L.Dropout(0.3))

model.add(L.Dense(16, activation='elu'))
model.add(L.BatchNormalization())
model.add(L.Dropout(0.4))

model.add(L.Dense(CLASS_NUM, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.0001), metrics=["accuracy"])
```
```python
history = model.fit(
        train_generator,
        steps_per_epoch = len(train_generator),
        validation_data = test_generator, 
        validation_steps = len(test_generator),
        epochs = 4,
        callbacks = callbacks_list
)
```
После этого увеличиваем количество обучаемых слоев и уменьшаем LR:
```python
base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
LR = 0.00005
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
```
```python
history = model.fit(
        train_generator,
        steps_per_epoch = len(train_generator),
        validation_data = test_generator, 
        validation_steps = len(test_generator),
        epochs = 5,
        callbacks = callbacks_list
)
```
Делаем так еще раз:
```python
base_model.trainable = True

fine_tune_at = 50

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
LR = 0.00001
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
```
```python
history = model.fit(
        train_generator,
        steps_per_epoch = len(train_generator),
        validation_data = test_generator, 
        validation_steps = len(test_generator),
        epochs = 5,
        callbacks = callbacks_list
)
```
Разблокируем все слои и доубачаем:
```python
base_model.trainable = True
LR = 0.000001
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=LR), metrics=["accuracy"])
```
```python
history = model.fit(
        train_generator,
        steps_per_epoch = len(train_generator)//5,
        validation_data = test_generator, 
        validation_steps = len(test_generator),
        epochs = 5,
        callbacks = callbacks_list
)
```
Test Time Augmentation (TTA) немного улучшает результат финального предсказания:
```python
test_sub_generator.reset()

tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1)
    predictions.append(preds)

pred = np.mean(predictions, axis=0)

predictions = np.argmax(pred, axis=-1) #multiple categories
label_map = (train_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
predictions = [label_map[k] for k in predictions]
```
Сохраняем наш сабмит и скачиваем его из ВМ:
```python
filenames_with_dir=test_sub_generator.filenames
submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
submission['Id'] = submission['Id'].replace('test_upload/','')
submission.to_csv('submission.csv', index=False)
print('Save submit')
```
```python
files.download('submission.csv') 
```
Финальная accuracy на kaggle:0.95745
