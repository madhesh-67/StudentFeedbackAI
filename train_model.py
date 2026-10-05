import os
import re
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/feedback_data.csv"
MODEL_DIR = "model"

MODEL_PATH = "model/feedback_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
ENCODER_PATH = "model/label_encoders.pkl"
HISTORY_PATH = "model/training_history.pkl"

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 128

BATCH_SIZE = 32
EPOCHS = 30

RANDOM_STATE = 42


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# CREATE MODEL FOLDER
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "feedback",
    "sentiment",
    "emotion",
    "aspect"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )

df = df.dropna()

df["feedback"] = df["feedback"].apply(
    clean_text
)

print(
    "Total records:",
    len(df)
)


# ============================================================
# LABEL ENCODING
# ============================================================

sentiment_encoder = LabelEncoder()
emotion_encoder = LabelEncoder()
aspect_encoder = LabelEncoder()


y_sentiment = sentiment_encoder.fit_transform(
    df["sentiment"]
)

y_emotion = emotion_encoder.fit_transform(
    df["emotion"]
)

y_aspect = aspect_encoder.fit_transform(
    df["aspect"]
)


print()
print("Sentiment classes:")
print(
    list(sentiment_encoder.classes_)
)

print()
print("Emotion classes:")
print(
    list(emotion_encoder.classes_)
)

print()
print("Aspect classes:")
print(
    list(aspect_encoder.classes_)
)


# ============================================================
# TOKENIZATION
# ============================================================

print()
print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    df["feedback"]
)

sequences = tokenizer.texts_to_sequences(
    df["feedback"]
)

X = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# SAVE TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


# ============================================================
# SAVE LABEL ENCODERS
# ============================================================

with open(
    ENCODER_PATH,
    "wb"
) as file:

    pickle.dump(
        {
            "sentiment": sentiment_encoder,
            "emotion": emotion_encoder,
            "aspect": aspect_encoder
        },
        file
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_sentiment
)


X_train = X[train_indices]
X_test = X[test_indices]

y_sentiment_train = y_sentiment[train_indices]
y_sentiment_test = y_sentiment[test_indices]

y_emotion_train = y_emotion[train_indices]
y_emotion_test = y_emotion[test_indices]

y_aspect_train = y_aspect[train_indices]
y_aspect_test = y_aspect[test_indices]


print()
print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# BUILD CNN + BIDIRECTIONAL LSTM MODEL
# ============================================================

print()
print("=" * 60)
print("BUILDING CNN + BiLSTM MODEL")
print("=" * 60)


inputs = Input(
    shape=(MAX_SEQUENCE_LENGTH,),
    name="feedback_input"
)


# Word Embedding
x = Embedding(
    input_dim=MAX_VOCAB_SIZE,
    output_dim=EMBEDDING_DIM,
    name="embedding"
)(inputs)


# CNN extracts local text features
x = Conv1D(
    filters=128,
    kernel_size=3,
    padding="same",
    activation="relu",
    name="cnn"
)(x)


x = MaxPooling1D(
    pool_size=2
)(x)


x = Dropout(
    0.25
)(x)


# Bidirectional LSTM captures context
x = Bidirectional(
    LSTM(
        64,
        dropout=0.2
    ),
    name="bidirectional_lstm"
)(x)


# Shared feature layer
x = Dense(
    64,
    activation="relu",
    name="shared_features"
)(x)


x = Dropout(
    0.30
)(x)


# ============================================================
# SENTIMENT OUTPUT
# ============================================================

sentiment_output = Dense(
    len(sentiment_encoder.classes_),
    activation="softmax",
    name="sentiment"
)(x)


# ============================================================
# EMOTION OUTPUT
# ============================================================

emotion_output = Dense(
    len(emotion_encoder.classes_),
    activation="softmax",
    name="emotion"
)(x)


# ============================================================
# ASPECT OUTPUT
# ============================================================

aspect_output = Dense(
    len(aspect_encoder.classes_),
    activation="softmax",
    name="aspect"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(
    inputs=inputs,
    outputs=[
        sentiment_output,
        emotion_output,
        aspect_output
    ]
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss={
        "sentiment":
            "sparse_categorical_crossentropy",

        "emotion":
            "sparse_categorical_crossentropy",

        "aspect":
            "sparse_categorical_crossentropy"
    },

    metrics={
        "sentiment": ["accuracy"],
        "emotion": ["accuracy"],
        "aspect": ["accuracy"]
    }
)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    X_train,

    {
        "sentiment": y_sentiment_train,
        "emotion": y_emotion_train,
        "aspect": y_aspect_train
    },

    validation_split=0.20,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST MODEL
# ============================================================

print()
print("=" * 60)
print("TESTING MODEL")
print("=" * 60)

predictions = model.predict(
    X_test,
    verbose=0
)


sentiment_predictions = np.argmax(
    predictions[0],
    axis=1
)

emotion_predictions = np.argmax(
    predictions[1],
    axis=1
)

aspect_predictions = np.argmax(
    predictions[2],
    axis=1
)


# ============================================================
# SENTIMENT REPORT
# ============================================================

print()
print("=" * 60)
print("SENTIMENT REPORT")
print("=" * 60)

print(
    classification_report(
        y_sentiment_test,
        sentiment_predictions,
        target_names=sentiment_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# EMOTION REPORT
# ============================================================

print()
print("=" * 60)
print("EMOTION REPORT")
print("=" * 60)

print(
    classification_report(
        y_emotion_test,
        emotion_predictions,
        target_names=emotion_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# ASPECT REPORT
# ============================================================

print()
print("=" * 60)
print("ASPECT REPORT")
print("=" * 60)

print(
    classification_report(
        y_aspect_test,
        aspect_predictions,
        target_names=aspect_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "wb"
) as file:

    pickle.dump(
        history.history,
        file
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Tokenizer:")
print(TOKENIZER_PATH)

print()
print("Label encoders:")
print(ENCODER_PATH)

print()
print("Training history:")
print(HISTORY_PATH)

print()