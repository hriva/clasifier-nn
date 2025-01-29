#!/usr/bin/env python3
from functools import partial
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential  # NN are sequential layers
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    GlobalMaxPooling1D,
    Dropout,
    LSTM,
    Bidirectional,
    LayerNormalization,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf


# data ingest #########################################################################
text_col = "text"
label_col = "label"
data = pd.read_parquet("data/data_001.parquet")

# OneHotEncoder #######################################################################
le = LabelBinarizer()
labels = le.fit_transform(data[label_col])  # np array of onehotencoded lables.
with open("data/label_list.ls", "w") as lbl:
    lbl.writelines(le.classes_ + "\n")

type(labels), labels.shape
# np.unique(labels)
num_classes = labels.shape[1]

classes_total = labels.sum(axis=0)
classes_total
classes_weights = {}
for i in range(0, len(classes_total)):
    classes_weights[i] = classes_total.max() / classes_total[i]
classes_weights

# Tokenize ###########################################################################
token_vector_max_lenght = 256
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  #  WordPiece: BERT

x_inputs = tokenizer(
    list(data[text_col]),
    truncation=True,
    padding="max_length",
    max_length=token_vector_max_lenght,
    return_attention_mask=True,
)
tk = partial(
    tokenizer.encode_plus,
    padding="max_length",
    truncation=True,
    max_length=token_vector_max_lenght,
    # return_tensors="tf",
)
# text_inputs = np.stack(data[text_col].apply(lambda x: tk(x)["input_ids"]).values)

# type(text_inputs), text_inputs.shape
type(x_inputs), x_inputs.keys()

text_inputs = np.array(x_inputs.get("input_ids"), dtype=int)
type(text_inputs), text_inputs.shape

# text_inputs = text_inputs.astype("int32")  # Convert to correct types
# labels = labels.astype("int32")

# train_test_split ###################################################################
X_train, X_test, y_train, y_test = train_test_split(
    text_inputs, labels, test_size=0.2, random_state=1, stratify=labels
)


# Helper function for class weights
def compute_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight

    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_integers), y=y_integers
    )
    return dict(enumerate(class_weights))


# Init NN ############################################################################
network = Sequential()

network.add(
    Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=384,  # Increased embedding dimension
        input_length=token_vector_max_lenght,
        mask_zero=True,  # Enable mask for variable length sequences
    )
)

# Add Bidirectional LSTM layers
network.add(Bidirectional(LSTM(256, return_sequences=True)))
network.add(LayerNormalization())
network.add(Dropout(0.2))

network.add(Bidirectional(LSTM(128)))
network.add(LayerNormalization())
network.add(Dropout(0.2))

# Dense layers with residual connections
network.add(Dense(256, activation="relu"))
network.add(LayerNormalization())
network.add(Dropout(0.4))

network.add(Dense(128, activation="relu"))
network.add(LayerNormalization())
network.add(Dropout(0.2))

# Output layer
network.add(Dense(num_classes, activation="softmax"))

# Compile the network
optimizer = Adam(learning_rate=2e-5)
network.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)
network.summary()

# Train ################################################################################
file_network = "text_clasificator.keras"
epochs = 50
batch_size = 24
checkpointer = ModelCheckpoint(
    file_network, monitor="val_loss", verbose=1, save_best_only=True
)


callbacks = [
    ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

history = network.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,  # Add this instead of steps_per_epoch
    epochs=epochs,
    callbacks=callbacks,
    class_weight=compute_class_weights(y_train),  # Add class weights if imbalanced
    verbose=1,
)

# test_size
predictions = network.predict(X_test, batch_size=batch_size)
pred_classes = le.classes_[np.argmax(predictions, axis=1)]
true_classes = le.classes_[np.argmax(y_test, axis=1)]

# Create a comparison DataFrame
results_df = pd.DataFrame({"Predicted": pred_classes, "Actual": true_classes})

# Show first few predictions
print("\nFirst 10 predictions:")
print(results_df.head(10))

# Get accuracy
accuracy = (pred_classes == true_classes).mean()
print(f"\nAccuracy: {accuracy:.2%}")

# dump model
network.save("models/network_1.keras")
# le.classes_
