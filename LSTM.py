import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# set the directory where the audio and transcript files are stored
data_dir = "dataset"

# define the size of the training and validation sets
train_set_size = 0.8  # 80% of the data will be used for training

# load the audio and transcript data
audio_files = []
transcripts = []

for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):
        # load the audio file
        audio_file = load_audio(os.path.join(data_dir, file_name))
        audio_files.append(audio_file)

        # load the transcript file
        transcript_file = load_transcript(os.path.join(data_dir, file_name.replace(".wav", ".txt")))
        transcripts.append(transcript_file)

# convert the transcript text to numerical values
chars = sorted(list(set("".join(transcripts))))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# create training and validation sets
split_index = int(len(audio_files) * train_set_size)
X_train = audio_files[:split_index]
Y_train = transcripts[:split_index]
X_valid = audio_files[split_index:]
Y_valid = transcripts[split_index:]

# prepare the data for training
n_chars = len(chars)
n_samples = len(X_train)

max_audio_length = max([len(audio) for audio in X_train])
max_transcript_length = max([len(transcript) for transcript in Y_train])

X_train = np.zeros((n_samples, max_audio_length, 1))
Y_train = np.zeros((n_samples, max_transcript_length, n_chars))

for i, (audio, transcript) in enumerate(zip(X_train, Y_train)):
    audio_len = len(audio_files[i])
    audio = audio_files[i].reshape(audio_len, 1)
    audio = audio / np.max(np.abs(audio))
    audio_pad = np.zeros((max_audio_length - audio_len, 1))
    audio = np.vstack((audio, audio_pad))
    X_train[i] = audio

    transcript_len = len(transcripts[i])
    for j, char in enumerate(transcripts[i]):
        Y_train[i, j, char_to_int[char]] = 1

# define the model architecture
model = Sequential()
model.add(LSTM(256, input_shape=(max_audio_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(n_chars)))
model.add(Activation("softmax"))

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# define the checkpoint to save the best model
checkpoint = ModelCheckpoint("best_model.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")

# train the model
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=100, batch_size=32, callbacks=[checkpoint])
