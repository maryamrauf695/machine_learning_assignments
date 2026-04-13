import os
import glob
import pickle
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────────
TRAINING_DIR = r"c:\Users\Ayesha Akmal\Downloads\archive (2)\Training"
IMG_SIZE     = 64
MAX_IMAGES   = 1000        # cap per class to avoid memory issues
MODEL_PATH   = "model.pkl"
# ────────────────────────────────────────────────────────────────────────────

def load_images(folder, label, max_images=None):
    images, labels = [], []
    files = glob.glob(os.path.join(folder, "*"))
    if max_images:
        files = files[:max_images]
    total = len(files)
    for i, path in enumerate(files, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten() / 255.0
        images.append(img)
        labels.append(label)
        if i % 500 == 0:
            print(f"  Loaded {i}/{total} from {os.path.basename(folder)}")
    return images, labels

print("Loading female images...")
female_imgs, female_labels = load_images(
    os.path.join(TRAINING_DIR, "female"), label=0, max_images=MAX_IMAGES
)

print("Loading male images...")
male_imgs, male_labels = load_images(
    os.path.join(TRAINING_DIR, "male"), label=1, max_images=MAX_IMAGES
)

X = np.array(female_imgs + male_imgs)
y = np.array(female_labels + male_labels)
print(f"\nDataset: {X.shape[0]} images  |  female={len(female_imgs)}  male={len(male_imgs)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Pipeline: Scaler + Logistic Regression ───────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
])

# ── Hyperparameter Grid ───────────────────────────────────────────────────────
param_grid = [
    {
        "clf__solver": ["saga"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    {
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
]

print("\nRunning GridSearchCV (3-fold CV) — this may take several minutes...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)

print(f"\nBest params : {grid_search.best_params_}")
print(f"Best CV acc : {grid_search.best_score_ * 100:.2f}%")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"\nBest model saved to {MODEL_PATH}")
