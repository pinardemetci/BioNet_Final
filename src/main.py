import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization,\
        Dropout, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from utils import *


def svm(Xtrain, ytrain, Xtest, ytest):
    ''' implement SVM classifier on 4 cell groups'''
    # reshape tensor
    Xtrain = np.reshape(Xtrain, (len(Xtrain), -1))
    Xtest = np.reshape(Xtest, (len(Xtest), -1))
    ytrain = np.apply_along_axis(lambda x: np.where(x == max(x))[0], axis=1, arr=ytrain)
    ytest = np.apply_along_axis(lambda x: np.where(x == max(x))[0], axis=1, arr=ytest)

    classifier = LinearSVC()
    classifier.fit(Xtrain, ytrain)
    return classifier.score(Xtest, ytest)


def knn(Xtrain, ytrain, Xtest, ytest):
    ''' implement kNN classifier on 4 cell groups'''
    # reshape tensor
    Xtrain = np.reshape(Xtrain, (len(Xtrain), -1))
    Xtest = np.reshape(Xtest, (len(Xtest), -1))
    ytrain = np.apply_along_axis(lambda x: np.where(x == max(x))[0], axis=1, arr=ytrain)
    ytest = np.apply_along_axis(lambda x: np.where(x == max(x))[0], axis=1, arr=ytest)

    classifier = KNeighborsClassifier()
    classifier.fit(Xtrain, ytrain)
    return classifier.score(Xtest, ytest)

def cnn_model():
    model = Sequential()
    f, stride = 32, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    model.add(BatchNormalization())
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    f, stride = 64, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    model.add(BatchNormalization())
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    f, stride = 128, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    model.add(BatchNormalization())
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    f, stride = 64, 3
    model.add(Conv2D(f, (stride, stride), activation='relu'))
    model.add(BatchNormalization())
    m_stride = 2
    model.add(MaxPooling2D((m_stride, m_stride)))

    d = 100
    cat = 4
    model.add(Flatten())
    model.add(Dense(d, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(cat, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def cnn_kernel():
    # implement CNN model from kernel 2
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=8, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=4, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=4, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    cat = 4
    model.add(Dense(cat, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def main():
    # get data
    X_train, y_train = get_data('../dataset2-master/images/TRAIN/')
    X_test, y_test = get_data('../dataset2-master/images/TEST/')
   # X_train, y_train = get_data('../dataset-master/augmented/TRAIN/')
   # X_test, y_test = get_data('../dataset-master/augmented/TEST/')
    #acc = svm(X_train, y_train, X_test, y_test)
    #acc = knn(X_train, y_train, X_test, y_test)
    #print(acc)
    # build model
    model = cnn_model() # cnn 1
    #model = cnn_kernel() # cnn 2
    filepath = "../results/weight_tr5.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
        verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, y_train, epochs=100, batch_size=1024,
        callbacks = callbacks_list, validation_data=(X_test, y_test))
    print(model.summary())
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Prediction score:", score)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)
    with open("./train_hist_original", "wb") as fname:
        pickle.dump(history.history, fname)
    np.savetxt("confusion_matrix_original.txt", confusion_mtx, fmt="%.4f")
    # plot results

if __name__ == "__main__":
    main()
