# Source from https://blog.francium.tech/build-your-own-image-classifier-with-tensorflow-and-keras-dc147a15e38e
# and https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data = './images/training'
test_data = './images/testing'

w = 128
h = 128

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(w, h, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    train_data,
    target_size=(w, h),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    test_data,
    target_size=(w, h),
    batch_size=32,
    class_mode='binary'
)

model.fit_generator(
    training_set,
    steps_per_epoch=100,  # 8000
    epochs=10,
    validation_data=test_set,
    validation_steps=800
)
model.save('doll_image_classification.h5')
model.summary()
