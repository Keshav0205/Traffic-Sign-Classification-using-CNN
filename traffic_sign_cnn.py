import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class traffic_sign_classifier():

    def __init__(self):

        with open('data0.pickle', 'rb') as file1:
            data = pickle.load(file1, encoding='latin1')

        #load data--------------------------------
        self.x_train = data['x_train'].transpose(0,2,3,1)
        self.y_train = data['y_train']
        self.x_valid = data['x_validation'].transpose(0,2,3,1)
        self.y_valid = data['y_validation']
        self.x_test = data['x_test'].transpose(0,2,3,1)
        self.y_test = data['y_test']
        self.labels = data['labels']
    #------------------------------------------

    def print_shape(self):
        print('x_train: {}'.format(self.x_train.shape))
        print('y_train: {}'.format(self.y_train.shape))
        print('x_valid: {}'.format(self.x_valid.shape))
        print('y_valid: {}'.format(self.y_valid.shape))
        print('x_test: {}'.format(self.x_test.shape))
        print('y_test: {}'.format(self.y_test.shape))
    
    def plot_grid(self, l_grid, w_grid):

        #plotting images and labels in a grid of 5 x 5:
        l_grid = 5
        w_grid = 5
        _, axes = plt.subplots(l_grid, w_grid, figsize = (10,10))
        axes = axes.ravel()
        for i in np.arange(0, l_grid * w_grid):
            j = np.random.randint(i, len(self.x_train))
            axes[i].imshow(self.x_train[j])
            axes[i].set_title(self.labels[self.y_train[j]], fontsize = 10)
            axes[i].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def gray_normalization(self):
        #shuffling datasets to prevent the NN to learn any possible sequence:
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        #grayscale_conversion:
        self.x_train_gray_norm = np.sum(self.x_train/3, axis = 3, keepdims=True)
        self.x_valid_gray_norm = np.sum(self.x_valid/3, axis = 3, keepdims=True)
        self.x_test_gray_norm = np.sum(self.x_test/3, axis = 3, keepdims=True)

        #normalization:-
        self.x_train_gray_norm = (self.x_train_gray_norm - 128) / 128
        self.x_valid_gray_norm = (self.x_valid_gray_norm - 128) / 128
        self.x_test_gray_norm = (self.x_test_gray_norm - 128) / 128

        #print('x_train_gray_norm: {}'.format(self.x_train_gray_norm.shape))
        #print('x_valid_gray_norm: {}'.format(self.x_valid_gray_norm.shape))
        #print('x_test_gray_norm: {}'.format(self.x_test_gray_norm.shape))
    
    def build_CNN_model(self, dropout = 0.2):
        self.CNN = Sequential([
            Conv2D(16, (3,3), padding = 'same', activation='relu', input_shape = (32,32,1)),
            MaxPooling2D((2,2)),
            Conv2D(32, (3,3), padding = 'same', activation='relu'),
            MaxPooling2D(2,2),
            Dropout(dropout),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(86,activation='relu'),
            Dense(43, activation='softmax')
        ])
        
    def save_model(self, save_path):
        self.CNN.save(save_path)
    
    def load_model_data(self, load_path):
        self.CNN = load_model(load_path)
    
    def compile_and_train(self, lr=0.001):
        self.CNN.compile(optimizer = Adam(learning_rate = lr), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        self.history = self.CNN.fit(self.x_train_gray_norm, self.y_train, batch_size = 500, epochs = 100,
                                    verbose = 3, validation_data = (self.x_valid_gray_norm, self.y_valid))
        self.score = self.CNN.evaluate(self.x_test_gray_norm, self.y_test)
    
    def plot_metrics(self):

        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']


        epochs = range(len(accuracy))
        plt.figure(1)
        plt.plot(epochs, accuracy, 'r-', label = 'Training accuracy')
        plt.plot(epochs, val_accuracy, 'b-', label = 'Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.show()

        plt.figure(2)
        plt.plot(epochs, loss, 'r-', label = 'Training loss')
        plt.plot(epochs, val_loss, 'b-', label = 'Validation loss')
        plt.title('Training and Validation loss')
        plt.show()
    
    def evaluate_model(self):
        score = self.CNN.evaluate(self.x_test_gray_norm, self.y_test, batch_size = 500)
        print(score)
     
    def test_plot(self):
        predicted_classes = self.CNN.predict_classes(self.x_test_gray_norm)
        l_grid = 2
        w_grid = 2
        _, axes = plt.subplots(l_grid, w_grid, figsize = (12,12))
        axes = axes.ravel()
        for i in np.arange(0, l_grid * w_grid):
            #j = np.random.randint(i, len(self.x_test))
            axes[i].imshow(self.x_test[i])
            # axes[i].set_title(self.labels[self.y_test[j]], fontsize = 8)
            print('Predicted_class: {} -- Label: {}'.format(predicted_classes[i], self.y_test[i]))
            axes[i].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.show()