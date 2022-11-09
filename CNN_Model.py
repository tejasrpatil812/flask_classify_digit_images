from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle

(trainX, trainY), (testX, testY) = mnist.load_data()

def preprocess_data():
    global trainX, testX, trainY, testY
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)) / 255.0
    testX = testX.reshape((testX.shape[0], 28, 28, 1)) / 255.0
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def augment_data():
    datagenerator = ImageDataGenerator(rotation_range=2, 
                                width_shift_range=0.1, 
                                height_shift_range=0.1, 
                                zoom_range=[1.01, 1.25])

    t_gen = datagenerator.flow(trainX, trainY, batch_size=256)
    v_gen=datagenerator.flow(testX, testY, batch_size=256)
    return t_gen, v_gen

def define_model():
    cnn_model=Sequential()
    cnn_model.add(Conv2D(32, (3,3) , activation='relu', input_shape=trainX.shape[1:])) 
    cnn_model.add(BatchNormalization())
    
    cnn_model.add(Conv2D(32, (3,3) , activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, (3,3) , activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, (3,3) , activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.5))
	
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.5))
	
    cnn_model.add(Dense(10, activation='softmax'))

    # model.summary()
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_model

def train_model():
    cnn_model = define_model()
    t_gen, v_gen = augment_data()
    early_stopper = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 
    his = cnn_model.fit_generator(t_gen, steps_per_epoch=175, 
                                epochs=20, validation_steps=20, 
                                validation_data=v_gen, 
                                callbacks=[early_stopper],
                                verbose = 1)
    _, acc = cnn_model.evaluate(testX, testY, verbose=1)
    filename = './new_model.sav'
    pickle.dump(cnn_model, open(filename, 'wb'))
    return his, acc

if __name__ == '__main__':
    preprocess_data()
    his, test_acc = train_model()
    print(f"Test Accuracy : {test_acc} \nHistory : {his}")
    