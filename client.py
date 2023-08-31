import os
import cv2
import random
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle


import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

NUM_CLIENTS = 2
# CLASSES_LIST = ["ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
#                 "BaseballPitch", "Basketball", "BenchPress", "Biking"]
CLASSES_LIST = os.listdir('data')
# CLASSES_LIST = CLASSES_LIST[1:41]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = "data"

model = Sequential([
    keras.layers.ConvLSTM2D(filters=2, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                            recurrent_dropout=0.2, return_sequences=True, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    keras.layers.MaxPooling3D(pool_size=(
        1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.TimeDistributed(keras.layers.Dropout(0.2)),
    keras.layers.ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh',
                            data_format="channels_last", recurrent_dropout=0.2, return_sequences=True),
    keras.layers.MaxPooling3D(pool_size=(
        1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.TimeDistributed(keras.layers.Dropout(0.2)),
    keras.layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh',
                            data_format="channels_last", recurrent_dropout=0.2, return_sequences=True),
    keras.layers.MaxPooling3D(pool_size=(
        1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.TimeDistributed(keras.layers.Dropout(0.2)),
    keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh',
                            data_format="channels_last", recurrent_dropout=0.2, return_sequences=True),
    keras.layers.MaxPooling3D(pool_size=(
        1, 2, 2), padding='same', data_format='channels_last'),
    keras.layers.Flatten(),
    keras.layers.Dense(len(CLASSES_LIST), activation="softmax")
])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


all_classes_names = os.listdir(
    'data')
random_range = random.sample(range(len(all_classes_names)), 2)

for counter, random_index in enumerate(random_range, 1):
    selected_class_Name = all_classes_names[random_index]
    video_files_names_list = os.listdir(
        f'data/{selected_class_Name}')
    selected_video_file_name = random.choice(video_files_names_list)
    video_reader = cv2.VideoCapture(
        f'data/{selected_class_Name}/{selected_video_file_name}')
    _, bgr_frame = video_reader.read()
    video_reader.release()
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cv2.putText(rgb_frame, selected_class_Name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    plt.subplot(5, 4, counter)
    plt.imshow(rgb_frame)
    plt.axis('off')


def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES,
                         frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    return tf.convert_to_tensor(frames_list, dtype=tf.float32)


def create_dataset():
    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = tf.convert_to_tensor(features, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return features, labels


client_accuracy_history = []
client_loss_history = []

features, labels = create_dataset()
# one_hot_encoded_labels = to_categorical(labels)

x_train = features
x_test = features
y_train = labels
y_test = labels
# x_train, x_test, y_train, y_test = train_test_split(features,test_size = 0.25, shuffle = True, random_state = seed_constant)

############


# Load dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# dist = [1,2]
# x_train, y_train = getData(dist, x_train, y_train)
# getDist(y_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client: get_parameters() called")
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
        r = model.fit(x_train, y_train, epochs=10, validation_data=(
            x_test, y_test), verbose=0, shuffle=True, batch_size=32,
            callbacks=[early_stopping_callback])

        hist = {
            'accuracy': r.history['accuracy'],
            'loss': r.history['loss']
        }

        avg_accuracy_per_round = np.mean(hist['accuracy'])
        avg_loss_per_round = np.mean(hist['loss'])

        print("Fit history : ", hist)

        client_accuracy_history.append(
            float(avg_accuracy_per_round))

        client_loss_history.append(
            float(avg_loss_per_round))

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8892",
    client=FlowerClient(),
    grpc_max_message_length=1024*1024*1024
)

history = {
    'accuracy': client_accuracy_history,
    'loss': client_loss_history
}
with open('client_loss_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print('Client Execution')
