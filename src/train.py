#!/usr/bin/env python3
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model  # NN are sequential layers
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
    LayerNormalization,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from src.lib import standardize_text


class NeuralNetwork:
    def __init__(self) -> None:
        self.file_network = "models/classifier.keras"
        self.token_vector_max_lenght = 256
        self.x_inputs = []
        self.__tokenizer = "bert-base-uncased"
        self.__autotokenizer = None
        self.batch_size = 32

        self.le = LabelBinarizer()

    def compile_network(self) -> None:
        self.network = Sequential()
        self.network.add(
            Embedding(
                input_dim=self.__autotokenizer.vocab_size,
                output_dim=384,  # Increased embedding dimension
                input_length=self.token_vector_max_lenght,
                mask_zero=True,  # Enable mask for variable length sequences
            )
        )

        # Add Bidirectional LSTM layers
        self.network.add(Bidirectional(LSTM(256, return_sequences=True)))
        self.network.add(LayerNormalization())
        self.network.add(Dropout(0.2))

        self.network.add(Bidirectional(LSTM(128)))
        self.network.add(LayerNormalization())
        self.network.add(Dropout(0.2))

        # Dense layers with residual connections
        self.network.add(Dense(256, activation="relu"))
        self.network.add(LayerNormalization())
        self.network.add(Dropout(0.4))

        self.network.add(Dense(128, activation="relu"))
        self.network.add(LayerNormalization())
        self.network.add(Dropout(0.2))

        # Output layer
        self.network.add(Dense(self.num_classes, activation="softmax"))

        # Compile the self.network
        self.network.compile(
            optimizer=Adam(learning_rate=2e-5),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        self.callbacks = [
            ModelCheckpoint(
                "Classifier-checkpoint.keras",
                monitor="val_accuracy",  # Instead of "val_loss"
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

    @property
    def tokenizer(self):
        return self.__tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: str = "bert-base-uncased") -> None:
        self.__tokenizer = tokenizer
        self.__autotokenizer = AutoTokenizer.from_pretrained(
            self.__tokenizer
        )  #  WordPiece: BERT

    def one_hot_encode(self, data: pd.DataFrame) -> None:
        self.labels = self.le.fit_transform(
            data["label"]
        )  # np array of onehotencoded lables.
        self.num_classes = self.labels.shape[1]

    def train_test_data(self, data: pd.DataFrame) -> None:
        self.tokenizer = "bert-base-uncased"
        self.x_inputs = self.__autotokenizer(
            list(data["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.token_vector_max_lenght,
            return_attention_mask=True,
        )
        self.text_inputs = np.array(self.x_inputs.get("input_ids"), dtype=int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.text_inputs,
            self.labels,
            test_size=0.2,
            random_state=1,
            stratify=self.labels,
        )

    def compute_class_weights(self, y):
        y_integers = np.argmax(y, axis=1)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_integers), y=y_integers
        )
        return dict(enumerate(class_weights))

    def save_model(self, output: str = "models/classifier.keras") -> None:
        self.network.save(output)
        with open(output + ".lb", "w") as lbl:
            lbl.writelines(self.le.classes_ + "\n")

    def train(self, batch_size: int = 24, epochs: int = 50) -> None:
        self.history = self.network.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            batch_size=self.batch_size,  # Add this instead of steps_per_epoch
            epochs=epochs,
            callbacks=self.callbacks,
            class_weight=self.compute_class_weights(
                self.y_train
            ),  # Add class weights if imbalanced
            verbose=1,
        )

    def load_model(self, model: str, labels: str) -> None:
        self.network = load_model(model)

        with open(labels, "r") as lbl:
            label_list = lbl.readlines()
        self.labels = [l.replace("\n", "") for l in label_list]

    def predict(self, vectors: List[str]) -> List[str]:
        if self.network is None:
            print("No model initialized")
            return

        try:
            if self.__autotokenizer is None:
                self.tokenizer = "bert-base-uncased"

            tokens = self.__autotokenizer(
                [standardize_text(x) for x in vectors],
                truncation=True,
                padding="max_length",
                max_length=self.token_vector_max_lenght,
                return_attention_mask=True,
            )
            text_vectors = np.array(tokens.get("input_ids"), dtype=int)
            predictions = self.network.predict(text_vectors, batch_size=self.batch_size)
            return self.labels[np.argmax(predictions, axis=1)[0]]
            # TODO: vector of resutls,
        except Exception as e:
            raise e
