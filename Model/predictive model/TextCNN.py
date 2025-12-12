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

#  float32，
tf.keras.backend.set_floatx('float32')

# ———— Kaggle  ————
RAW_DATA = '/kaggle/input/self-fixed-satd-dataset/raw_data_final_40501.json'
OUT_CV   = '/kaggle/working/textcnn_10.txt'
OUT_LOGO = '/kaggle/working/textcnn_logo.txt'

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



from tensorflow.keras.callbacks import EarlyStopping

# ———— 10  ————
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_metrics = {'auc':[], 'acc':[], 'prec':[], 'recall':[], 'f1':[]}

for fold, (tr_idx, te_idx) in enumerate(kf.split(data, labels), 1):
    model = build_textcnn()
    
    # ✳️ （）
    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,                # 3
        restore_best_weights=True, # 
        verbose=1
    )
    
    # ✳️  callbacks
    model.fit(
        data[tr_idx], labels[tr_idx],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stop]
    )
    
    y_proba = model.predict(data[te_idx], batch_size=BATCH_SIZE).ravel()
    y_pred  = (y_proba >= 0.5).astype(int)
    cv_metrics['auc'].append(roc_auc_score(labels[te_idx], y_proba))
    cv_metrics['acc'].append(accuracy_score(labels[te_idx], y_pred))
    cv_metrics['prec'].append(precision_score(labels[te_idx], y_pred, zero_division=0))
    cv_metrics['recall'].append(recall_score(labels[te_idx], y_pred, zero_division=0))
    cv_metrics['f1'].append(f1_score(labels[te_idx], y_pred, zero_division=0))
    print(f"Fold{fold}: AUC={cv_metrics['auc'][-1]:.4f}")

# ————  ————
with open(OUT_CV, 'w', encoding='utf-8') as f:
    for i in range(10):
        f.write(
            f"Fold{i+1}: AUC={cv_metrics['auc'][i]:.4f}, "
            f"Acc={cv_metrics['acc'][i]:.4f}, "
            f"Prec={cv_metrics['prec'][i]:.4f}, "
            f"Rec={cv_metrics['recall'][i]:.4f}, "
            f"F1={cv_metrics['f1'][i]:.4f}\n"
        )
    avg = {m: np.mean(v) for m, v in cv_metrics.items()}
    f.write(
        f": AUC={avg['auc']:.4f}, Acc={avg['acc']:.4f}, "
        f"Prec={avg['prec']:.4f}, Rec={avg['recall']:.4f}, F1={avg['f1']:.4f}\n"
    )
print("10CV", OUT_CV)


from tensorflow.keras.callbacks import EarlyStopping

# ————  (LOPO)  ————
logo_metrics = {'project':[], 'auc':[], 'acc':[], 'prec':[], 'recall':[], 'f1':[]}

for proj in np.unique(projects):
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
        f": AUC:{avg_logo['auc']:.4f}, "
        f"Acc:{avg_logo['acc']:.4f}, "
        f"Prec:{avg_logo['prec']:.4f}, "
        f"Rec:{avg_logo['recall']:.4f}, "
        f"F1:{avg_logo['f1']:.4f}\n"
    )
print("", OUT_LOGO)