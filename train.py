import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import random
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from fpdf import FPDF

# === CONFIGURATION ===
print("Step 1: Configuration")
xml_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/labelXmls"
tif_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/images"
img_size = (224, 224)
batch_size = 32
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# === PARSE DATA ===
print("Step 2: Parsing XML files")
all_data = []
all_labels = []

for file in sorted(os.listdir(xml_dir)):
    if file.endswith(".xml"):
        base = file.split(".")[0]
        xml_path = os.path.join(xml_dir, file)
        tif_path = os.path.join(tif_dir, f"{base}.tif")
        if not os.path.exists(tif_path): continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall(".//object"):
                label = obj.find(".//possibleresult/name").text
                points = obj.findall(".//point")
                coords = [tuple(map(float, pt.text.split(","))) for pt in points]
                if not coords: continue
                xs, ys = zip(*coords)
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))
                all_data.append((tif_path, (xmin, ymin, xmax, ymax)))
                all_labels.append(label)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

print(f"Total object instances found: {len(all_data)}")

# === ENCODE LABELS ===
print("Step 3: Encoding labels")
le = LabelEncoder()
encoded_labels = le.fit_transform(all_labels)
num_classes = len(le.classes_)

# === SPLIT DATA ===
print("Step 4: Splitting into train and validation sets")
combined = list(zip(all_data, encoded_labels))
random.shuffle(combined)
split_idx = int(0.8 * len(combined))
train_data = combined[:split_idx]
val_data = combined[split_idx:]

# === GENERATOR ===
print("Step 5: Creating data generator")
class FAIR1MGenerator(Sequence):
    def __init__(self, data, batch_size, is_train=True):
        self.data = data
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for (img_path, (xmin, ymin, xmax, ymax)), label in batch:
            try:
                img = Image.open(img_path).convert("RGB")
                crop = img.crop((xmin, ymin, xmax, ymax)).resize(img_size)
                X.append(np.array(crop, dtype=np.float32) / 255.0)
                y.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        return np.array(X), to_categorical(y, num_classes)

# === MODEL ===
print("Step 6: Building CNN model")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAINING ===
print("Step 7: Training model with generator")
train_gen = FAIR1MGenerator(train_data, batch_size)
val_gen = FAIR1MGenerator(val_data, batch_size, is_train=False)

history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# === SAVE MODEL ===
model_save_path = os.path.join(results_dir, "fair1m_cnn_model.h5")
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")

# === PLOTS ===
print("Step 8: Generating accuracy curve")
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)
acc_path = os.path.join(results_dir, "accuracy_curve.png")
plt.savefig(acc_path)
plt.close()

# === EVALUATION ===
print("Step 9: Evaluating on validation set")
X_val, y_val = val_gen.__getitem__(0)
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
label_names = le.inverse_transform(unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
cm_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# === PDF REPORT ===
print("Step 10: Saving PDF report")
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "CN7023 FAIR1M Deep Learning Report", ln=True, align="C")
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Training Results Summary", ln=True)
pdf.set_font("Arial", "", 12)
pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
pdf.cell(0, 10, f"Train Accuracy: {round(history.history['accuracy'][-1] * 100, 2)}%", ln=True)
pdf.cell(0, 10, f"Val Accuracy: {round(history.history['val_accuracy'][-1] * 100, 2)}%", ln=True)
pdf.ln(10)

pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Accuracy Curve:", ln=True)
pdf.image(acc_path, w=180)
pdf.ln(10)

pdf.cell(0, 10, "Confusion Matrix:", ln=True)
pdf.image(cm_path, w=180)
pdf.ln(10)

pdf.output(os.path.join(results_dir, "FAIR1M_Results_Report.pdf"))
print("✅ PDF report saved!")

