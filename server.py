import time
import pickle
from ast import Dict, Tuple
import flwr as fl
import sys
import numpy as np
import os
import cv2
import random
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard

import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple
from flwr import common

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

NUM_CLIENTS = 2
# CLASSES_LIST = ["ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
#                 "BaseballPitch", "Basketball", "BenchPress", "Biking"]
CLASSES_LIST = os.listdir('data')
# CLASSES_LIST = CLASSES_LIST[1:5]
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

all_classes_names = os.listdir('data')
random_range = random.sample(range(len(all_classes_names)), 5)

for counter, random_index in enumerate(random_range, 1):
    selected_class_Name = all_classes_names[random_index]
    video_files_names_list = os.listdir(f'data/{selected_class_Name}')
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

    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)

    features = tf.convert_to_tensor(features, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return features, labels


features, labels = create_dataset()

x_train = features
x_test = features
y_train = labels
y_test = labels

early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir="logs")

convlstm_model_training_history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=32,
                                            shuffle=True, validation_split=0.2,
                                            callbacks=[early_stopping_callback, tensorboard_callback])


def get_evaluate_fn(model, x_val, y_val):
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}
    return evaluate


def fit_config(server_round: int):

    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }

    return config


def evaluate_config(server_round: int):
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,
    fraction_evaluate=0.2,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_fn=get_evaluate_fn(model, x_test, y_test),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
)

fl.server.start_server(
    server_address="0.0.0.0:8892",
    config=fl.server.ServerConfig(num_rounds=5),
    grpc_max_message_length=1024*1024*1024,
    strategy=strategy
)

print('FL Completed')

server_accuracy_history = []
server_loss_history = []

for r in convlstm_model_training_history.history['accuracy']:
    server_accuracy_history.append(r)
for r in convlstm_model_training_history.history['loss']:
    server_loss_history.append(r)

while not os.path.exists('client_loss_history.pkl'):
    print("Waiting for the file...")
    time.sleep(1)  # Adjust the sleep duration as needed
    print("Waiting for the file completed")


with open('client_loss_history.pkl', 'rb') as f:
    client_history = pickle.load(f)

server_accuracy_history.extend(client_history['accuracy'])
server_loss_history.extend(client_history['loss'])


# Create plots for accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(server_accuracy_history, label='Server Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Rounds and Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(server_loss_history, label='Server Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Rounds and Epochs')
plt.legend()

plt.tight_layout()
plt.show()
