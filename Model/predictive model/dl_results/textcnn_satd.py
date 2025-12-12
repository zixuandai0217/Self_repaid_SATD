import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping

#  float32，
tf.keras.backend.set_floatx('float32')

# ———— Kaggle  ————
RAW_DATA = 'raw_data_final_40501.json'
OUT_CV   = 'textcnn_10.txt'
OUT_LOGO = 'textcnn_logo.txt'

# ———— TextCNN  ————
MAX_WORDS = 20000
MAX_LEN   = 100
EMB_DIM   = 300
FILTER_SIZES = (1, 2, 3, 4, 5)
NUM_FILTERS  = 200
DROPOUT_RATE = 0.5
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 50

# ————  ————
with open(RAW_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)
texts    = df['f_comment'].astype(str).tolist()
labels   = df['is_self_fixed'].astype(int).values
projects = df['project_name'].astype(str).tolist()

# ————  ————
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_LEN)

# ———— TextCNN  ————
def build_textcnn():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=MAX_WORDS, output_dim=EMB_DIM)(inp)
    convs = []
    for sz in FILTER_SIZES:
        c = Conv1D(filters=NUM_FILTERS, kernel_size=sz, activation='relu')(x)
        p = GlobalMaxPooling1D()(c)
        convs.append(p)
    x = Concatenate()(convs)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy')
    return model

print("preprocessing Done...")


# ————  (LOPO)  ————
logo_metrics = {'project': [], 'auc': [], 'acc': [], 'prec': [], 'recall': [], 'f1': []}

# 
unique_projects = np.unique(projects)

# ✳️ 5050
selected_projects = np.concatenate([
    unique_projects[:50],    # 50
    # unique_projects[-55:]    # 55
])

print(f" {len(selected_projects)} LOPO")

for proj in selected_projects:
    idx_tr = np.where(np.array(projects) != proj)[0]
    idx_te = np.where(np.array(projects) == proj)[0]
    X_tr, y_tr = data[idx_tr], labels[idx_tr]
    X_te, y_te = data[idx_te], labels[idx_te]
    model = build_textcnn()

    # ✳️ （）
    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,                # 3
        restore_best_weights=True, # 
        verbose=1
    )

    # ✳️ fitcallbacks
    model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stop]
    )

    y_proba = model.predict(X_te, batch_size=BATCH_SIZE).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    logo_metrics['project'].append(proj)
    logo_metrics['auc'].append(roc_auc_score(y_te, y_proba))
    logo_metrics['acc'].append(accuracy_score(y_te, y_pred))
    logo_metrics['prec'].append(precision_score(y_te, y_pred, zero_division=0))
    logo_metrics['recall'].append(recall_score(y_te, y_pred, zero_division=0))
    logo_metrics['f1'].append(f1_score(y_te, y_pred, zero_division=0))
    print(f"Proj:{proj} AUC={logo_metrics['auc'][-1]:.4f}")
    safe_proj = proj.replace('/', '_').replace('\\', '_')  #  Windows  Linux
    with open(f'{safe_proj}.txt', 'w', encoding='utf-8') as f:
        f.write(
            f"Proj{proj}: "
            f"AUC:{roc_auc_score(y_te, y_proba):.4f}, "
            f"Acc:{accuracy_score(y_te, y_pred):.4f}, "
            f"Prec:{precision_score(y_te, y_pred, zero_division=0):.4f}, "
            f"Rec:{recall_score(y_te, y_pred, zero_division=0):.4f}, "
            f"F1:{f1_score(y_te, y_pred, zero_division=0):.4f}\n"
        )

# 
with open(OUT_LOGO, 'w', encoding='utf-8') as f:
    for i, proj in enumerate(logo_metrics['project']):
        f.write(
            f"Proj{proj}: "
            f"AUC:{logo_metrics['auc'][i]:.4f}, "
            f"Acc:{logo_metrics['acc'][i]:.4f}, "
            f"Prec:{logo_metrics['prec'][i]:.4f}, "
            f"Rec:{logo_metrics['recall'][i]:.4f}, "
            f"F1:{logo_metrics['f1'][i]:.4f}\n"
        )
    avg_logo = {m: np.mean(v) for m, v in logo_metrics.items() if m != 'project'}
    f.write(
        f"(50): "
        f"AUC:{avg_logo['auc']:.4f}, "
        f"Acc:{avg_logo['acc']:.4f}, "
        f"Prec:{avg_logo['prec']:.4f}, "
        f"Rec:{avg_logo['recall']:.4f}, "
        f"F1:{avg_logo['f1']:.4f}\n"
    )

print("", OUT_LOGO)