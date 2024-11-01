import os  # Importar el módulo para operaciones del sistema
import numpy as np  # Importar NumPy para operaciones matemáticas y arreglos
import pandas as pd  # Importar Pandas para manipulación de datos
from shutil import copyfile  # Importar función para copiar archivos
import matplotlib.pyplot as plt  # Importar Matplotlib para graficar

import tensorflow as tf  # Importar TensorFlow para aprovechar la GPU

# Configuración para la GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from keras import backend as K  # Importar backend de Keras
from keras import layers as L  # Importar capas de Keras
from keras.models import Model, load_model  # Importar modelo y función para cargar modelos
from keras.regularizers import l2  # Importar regularizador L2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping  # Importar callbacks para el entrenamiento
from tensorflow.keras.preprocessing import image  # Importar funciones de preprocesamiento de imágenes de Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importar generador de datos de imágenes
'''

# Crear un directorio para almacenar las imágenes
os.mkdir('./images/')
alpha = 'a'  # Inicializar la letra para los directorios
for i in range(0, 26):  # Crear directorios para cada letra de la A a la Z
    os.mkdir('./images/' + alpha)
    alpha = chr(ord(alpha) + 1)  # Pasar a la siguiente letra

rootdir = 'Braille_Dataset/'  # Directorio raíz del conjunto de datos
for file in os.listdir(rootdir):  # Recorrer los archivos en el directorio
    letter = file[0]  # Obtener la letra del archivo
    copyfile(rootdir + file, './images/' + letter + '/' + file)  # Copiar el archivo a su respectivo directorio
'''
# Inicializar el generador de datos de imágenes con aumentos
datagen = ImageDataGenerator(rotation_range=20,
                             shear_range=10,
                             validation_split=0.2)

# Crear un generador para el conjunto de entrenamiento
train_generator = datagen.flow_from_directory('./images/',
                                              target_size=(28, 28),
                                              subset='training')

# Crear un generador para el conjunto de validación
val_generator = datagen.flow_from_directory('./images/',
                                            target_size=(28, 28),
                                            subset='validation')

K.clear_session()  # Limpiar la sesión de Keras para evitar problemas de memoria

# Configurar los callbacks para el entrenamiento
model_ckpt = ModelCheckpoint('BrailleNet.keras', save_best_only=True)  # Guardar el mejor modelo
reduce_lr = ReduceLROnPlateau(patience=8, verbose=0)  # Reducir la tasa de aprendizaje si no hay mejora
early_stop = EarlyStopping(patience=15, verbose=1)  # Detener el entrenamiento si no hay mejora

# Definir la arquitectura del modelo
entry = L.Input(shape=(28, 28, 3))  # Entrada del modelo
x = L.SeparableConv2D(64, (3, 3), activation='relu')(entry)  # Primera capa convolucional separable
x = L.MaxPooling2D((2, 2))(x)  # Capa de max pooling
x = L.SeparableConv2D(128, (3, 3), activation='relu')(x)  # Segunda capa convolucional separable
x = L.MaxPooling2D((2, 2))(x)  # Capa de max pooling
x = L.SeparableConv2D(256, (2, 2), activation='relu')(x)  # Tercera capa convolucional separable
x = L.GlobalMaxPooling2D()(x)  # Capa de max pooling global
x = L.Dense(256)(x)  # Capa densa
x = L.LeakyReLU()(x)  # Activación LeakyReLU
x = L.Dense(64, kernel_regularizer=l2(2e-4))(x)  # Capa densa con regularización L2
x = L.LeakyReLU()(x)  # Activación LeakyReLU
x = L.Dense(26, activation='softmax')(x)  # Capa de salida con activación softmax

# Crear el modelo
model = Model(entry, x)
# Compilar el modelo con función de pérdida y optimizador
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=666,  # Número de épocas
    callbacks=[model_ckpt, reduce_lr, early_stop],  # Callbacks
    verbose=1  # Mostrar información del entrenamiento
)

# Cargar el mejor modelo guardado
model = load_model('BrailleNet.keras')
acc = model.evaluate(val_generator)[1]  # Evaluar el modelo en el conjunto de validación
print('Precisión del modelo: {}'.format(round(acc, 4)))  # Mostrar la precisión del modelo

# Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='pérdida de validación')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')  # Guardar la gráfica de pérdida

# Graficar la precisión durante el entrenamiento
plt.plot(history.history['accuracy'], label='precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='precisión de validación')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')  # Guardar la gráfica de precisión

# Realizar predicciones en el conjunto de validación
pred = model.predict(val_generator)
pred

# Cargar y mostrar una imagen
img = image.load_img('letra.png')  # Cargar la imagen
plt.imshow(img)  # Mostrar la imagen

x = image.img_to_array(img)  # Convertir la imagen a un array
x = np.expand_dims(x, axis=0)  # Expandir dimensiones para la predicción
model.predict(x)  # Realizar la predicción

a = np.argmax(model.predict(x), axis=1)  # Obtener la clase predicha
print('predicción = ', a)  # Mostrar la clase predicha
