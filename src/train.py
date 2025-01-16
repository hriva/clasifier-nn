from functools import partial
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential  # NN are sequential layers
from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPooling1D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import tensorflow

tensorflow.__version__

# data ingest #########################################################################
text_col = "Descripción de la mercancía"
label_col = "Tarifario"
data = pd.read_parquet("data/data.parquet")


# OneHotEncoder #######################################################################
le = LabelBinarizer()
labels = le.fit_transform(data[label_col])  # np array of onehotencoded lables.
labels.dump("models/labels.keras")
type(labels), labels.shape
# np.unique(labels)
num_classes = labels.shape[1]

# Tokenize ############################################################################
token_vector_max_lenght = 256
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  #  WordPiece: BERT
# # Tokenize the text data
# Load a pre-trained transformer model (e.g., DistilBERT for text classification)
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
# train_encodings = tokenizer(list(df['text_column']), truncation=True, padding=True, max_length=256)
# val_encodings = tokenizer(list(df['val_text_column']), truncation=True, padding=True, max_length=256)
tk = partial(
    tokenizer.encode_plus,
    padding="max_length",
    truncation=True,
    max_length=token_vector_max_lenght,
    # return_tensors="tf",
)
text_inputs = np.stack(data[text_col].apply(lambda x: tk(x)["input_ids"]).values)

type(text_inputs), text_inputs.shape
text_inputs = text_inputs.astype("int32")  # Convert to correct types
labels = labels.astype("int32")

# train_test_split ###################################################################
X_train, X_test, y_train, y_test = train_test_split(
    text_inputs, labels, test_size=0.2, random_state=1, stratify=labels
)

# Init NN ############################################################################
network = Sequential()

# Embedding Layer
network.add(
    Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=128,
        input_length=token_vector_max_lenght,
    )
)

# Dense Hidden Layers with Dropout
network.add(GlobalMaxPooling1D())  # Reduce dimension after embeddings
network.add(Dense(128, activation="relu"))
network.add(Dropout(0.5))
network.add(Dense(64, activation="relu"))
network.add(Dropout(0.5))

# Output Layer
# Softmax is needed for more than 2 classes
network.add(Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid"))


# Compile the network
optimizer = Adam(learning_rate=3e-5)
network.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
    metrics=["accuracy"],
)
network.summary()

# Train ################################################################################
file_network = "text_clasificator.keras"
epochs = 20
batch_size = 32
checkpointer = ModelCheckpoint(
    file_network, monitor="val_loss", verbose=1, save_best_only=True
)

history = network.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,  # Add this instead of steps_per_epoch
    epochs=epochs,
    # class_weight=classes_weights,
    verbose=1,
    callbacks=[checkpointer],
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
network.save("models/network.keras")
