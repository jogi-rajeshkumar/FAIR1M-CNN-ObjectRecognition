# import os
# import numpy as np
# from PIL import Image
# import xml.etree.ElementTree as ET
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # CONFIG
# model_path = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/results/fair1m_cnn_model.h5"
# xml_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/labelXmls"
# tif_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/images"
# img_size = (224, 224)
# results_file = "./results/fair1m_test_predictions.txt"

# # Load Model
# model = load_model(model_path)

# # Load validation samples
# print("Loading validation data...")
# all_data, all_labels = [], []

# for file in sorted(os.listdir(xml_dir)):
#     if not file.endswith(".xml"): continue
#     base = file.split(".")[0]
#     xml_path = os.path.join(xml_dir, file)
#     tif_path = os.path.join(tif_dir, base + ".tif")
#     if not os.path.exists(tif_path): continue

#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for obj in root.findall(".//object"):
#             label = obj.find(".//possibleresult/name").text
#             points = obj.findall(".//point")
#             coords = [tuple(map(float, pt.text.split(","))) for pt in points]
#             if not coords: continue
#             xs, ys = zip(*coords)
#             xmin, xmax = int(min(xs)), int(max(xs))
#             ymin, ymax = int(min(ys)), int(max(ys))
#             all_data.append((tif_path, (xmin, ymin, xmax, ymax), base))
#             all_labels.append(label)
#     except Exception as e:
#         print(f"Error parsing {file}: {e}")

# # Encode labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(all_labels)

# # Select 50 samples
# sample_data = all_data[:50]
# sample_labels = y_encoded[:50]
# X_test, sources = [], []

# for (img_path, (xmin, ymin, xmax, ymax), src_name) in sample_data:
#     img = Image.open(img_path).convert("RGB")
#     crop = img.crop((xmin, ymin, xmax, ymax)).resize(img_size)
#     X_test.append(np.array(crop, dtype=np.float32) / 255.0)
#     sources.append(src_name)

# X_test = np.array(X_test)
# y_true = sample_labels[:len(X_test)]

# # Predict
# print("Running predictions...")
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Save results
# with open(results_file, "w") as f:
#     f.write("FAIR1M Test Predictions (50 Samples)\\n")
#     f.write("===================================\\n\\n")
#     for i in range(len(X_test)):
#         true_label = le.inverse_transform([y_true[i]])[0]
#         pred_label = le.inverse_transform([y_pred[i]])[0]
#         f.write(f"Image {i+1:02d} ({sources[i]}): TRUE = {true_label}, PREDICTED = {pred_label}\\n")

# print(f"✅ Saved predictions to {results_file}")

#________________________________________________________________________________________

# import os
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import xml.etree.ElementTree as ET
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # CONFIG
# model_path = "./results/fair1m_cnn_model.h5"
# xml_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/labelXmls"
# tif_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/images"
# img_size = (224, 224)
# save_dir = "./results/predicted_images"
# os.makedirs(save_dir, exist_ok=True)

# # Load model
# model = load_model(model_path)

# # Load data
# all_data, all_labels = [], []
# for file in sorted(os.listdir(xml_dir)):
#     if not file.endswith(".xml"): continue
#     base = file.split(".")[0]
#     xml_path = os.path.join(xml_dir, file)
#     tif_path = os.path.join(tif_dir, base + ".tif")
#     if not os.path.exists(tif_path): continue

#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for obj in root.findall(".//object"):
#             label = obj.find(".//possibleresult/name").text
#             points = obj.findall(".//point")
#             coords = [tuple(map(float, pt.text.split(","))) for pt in points]
#             if not coords: continue
#             xs, ys = zip(*coords)
#             xmin, xmax = int(min(xs)), int(max(xs))
#             ymin, ymax = int(min(ys)), int(max(ys))
#             all_data.append((tif_path, (xmin, ymin, xmax, ymax), base, coords))
#             all_labels.append(label)
#     except Exception as e:
#         print(f"Error parsing {file}: {e}")

# # Encode labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(all_labels)

# # # Take first 50 samples
# # sample_data = all_data[:50]
# # sample_labels = y_encoded[:50]

# # Predict and plot
# for idx, (img_path, box, base_name, polygon) in enumerate(sample_data):
#     img = Image.open(img_path).convert("RGB")
#     crop = img.crop(box).resize(img_size)
#     X_input = np.array(crop, dtype=np.float32) / 255.0
#     X_input = np.expand_dims(X_input, axis=0)
#     y_pred = model.predict(X_input)
#     pred_label = le.inverse_transform([np.argmax(y_pred)])[0]
    
#     draw = ImageDraw.Draw(img)
#     draw.line(polygon + [polygon[0]], fill="red", width=2)
#     draw.text((box[0], box[1] - 10), f"Pred: {pred_label}", fill="red")
    
#     save_path = os.path.join(save_dir, f"prediction_{idx+1:02d}_{base_name}.jpg")
#     img.save(save_path)

# print(f"✅ Saved 50 predicted images to {save_dir}")

#________________________________________________________________________________________


# import os
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import xml.etree.ElementTree as ET
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # CONFIG
# model_path = "./results/fair1m_cnn_model.h5"
# xml_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/labelXmls"
# tif_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/images"
# img_size = (224, 224)
# save_dir = "./results/predicted_images_all"
# os.makedirs(save_dir, exist_ok=True)

# # Load model
# model = load_model(model_path)

# # Load data
# all_data, all_labels = [], []
# for file in sorted(os.listdir(xml_dir)):
#     if not file.endswith(".xml"): continue
#     base = file.split(".")[0]
#     xml_path = os.path.join(xml_dir, file)
#     tif_path = os.path.join(tif_dir, base + ".tif")
#     if not os.path.exists(tif_path): continue

#     try:
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for obj in root.findall(".//object"):
#             label = obj.find(".//possibleresult/name").text
#             points = obj.findall(".//point")
#             coords = [tuple(map(float, pt.text.split(","))) for pt in points]
#             if not coords: continue
#             xs, ys = zip(*coords)
#             xmin, xmax = int(min(xs)), int(max(xs))
#             ymin, ymax = int(min(ys)), int(max(ys))
#             all_data.append((tif_path, (xmin, ymin, xmax, ymax), base, coords))
#             all_labels.append(label)
#     except Exception as e:
#         print(f"Error parsing {file}: {e}")

# # Encode labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(all_labels)

# # Predict and plot all
# print(f"Generating predictions for {len(all_data)} objects...")
# for idx, (img_path, box, base_name, polygon) in enumerate(all_data):
#     try:
#         img = Image.open(img_path).convert("RGB")
#         crop = img.crop(box).resize(img_size)
#         X_input = np.array(crop, dtype=np.float32) / 255.0
#         X_input = np.expand_dims(X_input, axis=0)
#         y_pred = model.predict(X_input)
#         pred_label = le.inverse_transform([np.argmax(y_pred)])[0]
        
#         font = ImageFont.truetype("DejaVuSans.ttf", size=24)

#         draw = ImageDraw.Draw(img)
#         draw.line(polygon + [polygon[0]], fill="red", width=2)
#         draw.text((box[0], box[1] - 10), f"Pred: {pred_label}", fill="red", font=font)

#         # draw = ImageDraw.Draw(img)
#         # draw.line(polygon + [polygon[0]], fill="red", width=2)
#         # draw.text((box[0], box[1] - 10), f"Pred: {pred_label}", fill="red")

#         save_path = os.path.join(save_dir, f"prediction_{idx+1:05d}_{base_name}.jpg")
#         img.save(save_path)

#         if idx % 100 == 0:
#             print(f"Saved {idx+1}/{len(all_data)} images...")
#     except Exception as e:
#         print(f"Failed on {img_path}: {e}")

# print(f"✅ Done. Saved all predicted images to {save_dir}")


import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# CONFIG
model_path = "./results/fair1m_cnn_model.h5"
xml_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/labelXmls"
tif_dir = "/home/rajeshkumarjogi/Desktop/CN7023/Chinnu/FAIR1M/data/images"
img_size = (224, 224)
save_dir = "./results/predicted_combined_images"
os.makedirs(save_dir, exist_ok=True)

# Load model
model = load_model(model_path)

# Load and encode labels
all_labels = []
label_lookup = {}

# First pass to get labels
for file in sorted(os.listdir(xml_dir)):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall(".//object"):
                label = obj.find(".//possibleresult/name").text
                all_labels.append(label)
        except:
            continue

le = LabelEncoder()
le.fit(all_labels)

# Font
try:
    font = ImageFont.truetype("DejaVuSans.ttf", size=20)
except:
    font = ImageFont.load_default()

# Process each image
print("Processing all images...")
for file in sorted(os.listdir(xml_dir)):
    if not file.endswith(".xml"): continue
    base = file.split(".")[0]
    xml_path = os.path.join(xml_dir, file)
    tif_path = os.path.join(tif_dir, base + ".tif")
    if not os.path.exists(tif_path): continue

    try:
        img = Image.open(tif_path).convert("RGB")
        draw = ImageDraw.Draw(img)
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
            crop = img.crop((xmin, ymin, xmax, ymax)).resize(img_size)
            X_input = np.expand_dims(np.array(crop, dtype=np.float32) / 255.0, axis=0)
            y_pred = model.predict(X_input)
            pred_label = le.inverse_transform([np.argmax(y_pred)])[0]
            draw.line(coords + [coords[0]], fill="red", width=2)
            draw.text((xmin, ymin - 10), f"{pred_label}", fill="red", font=font)

        save_path = os.path.join(save_dir, f"{base}_predicted.jpg")
        img.save(save_path)
        print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("✅ Done. All combined prediction images saved.")