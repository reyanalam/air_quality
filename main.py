import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2

train_data = ImageDataGenerator(
                                rescale=1./255, #Normalizing
                                validation_split=0.2 #setting 20% for validation
                                   )

train_generator = train_data.flow_from_directory(
    r'C:\Users\Reyan\Desktop\Projects\Image_classification\data\train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training')


validation_generator = train_data.flow_from_directory(
    r'C:\Users\Reyan\Desktop\Projects\Image_classification\data\train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

model = Sequential([
    # Convolutional layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3),kernel_regularizer=l2(0.1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional layer 2
    Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional layer 3
    Conv2D(128, (3, 3), activation='relu',kernel_regularizer=l2(0.1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flattening the output of convolutional layers to feed it to the dense layer
    Flatten(),
    
    # Output layer for 5 classes
    Dense(5, activation='softmax')
])
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100)

model.save('image_classifier.h5')