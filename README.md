# FAIR1M-CNN-ObjectRecognition

🚀 A memory-efficient deep learning pipeline for fine-grained object recognition on the FAIR1M satellite dataset using a custom Keras data generator and lightweight CNN architecture.

---

## 📌 Project Overview

This project addresses the challenge of recognizing fine-grained objects in high-resolution satellite imagery using the **FAIR1M dataset**. It implements a **custom Keras data generator** to train a **CNN model** on modest hardware by dynamically loading image patches.

### 🔍 Key Features
- Custom `FAIR1MGenerator` for batch-wise TIFF+XML parsing
- Lightweight CNN with 71.04% training and 66.61% validation accuracy
- Full memory-efficient pipeline (runs on CPU-only machines)
- Annotated prediction visualizations and confusion matrix

---

## 🧠 Model Architecture

The CNN uses 3 convolutional layers with increasing filter depth, followed by a dense layer and dropout for regularization.

```
Input (224x224x3)
→ Conv2D (32 filters) + ReLU + MaxPooling
→ Conv2D (64 filters) + ReLU + MaxPooling
→ Conv2D (128 filters) + ReLU + MaxPooling
→ Flatten → Dense(128) + ReLU → Dropout(0.5)
→ Dense(37) + Softmax
```

---

## 🗂 Dataset: FAIR1M

- Source: [FAIR1M on Hugging Face](https://huggingface.co/papers/2103.05569)
- 15,000 high-resolution TIFF images
- Over 1 million object annotations across 37 subcategories

### 🖼️ FAIR1M Sample Images

![FAIR1M Sample 1](images/100.jpg)
![FAIR1M Sample 2](images/1036.jpg)
![FAIR1M Sample 3](images/1054.jpg)

---

## 📈 Results

### 🎯 Accuracy
- **Training Accuracy**: 71.04%
- **Validation Accuracy**: 66.61%

### 📊 Accuracy Curve

![Accuracy Plot](results/accuracy_curve.png)

### 🔁 Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### 📸 Predicted Output Visualizations

![Sample Output 1](results/predicted_combined_images/100_predicted.jpg)
![Sample Output 2](results/predicted_combined_images/1036_predicted.jpg)
![Sample Output 3](results/predicted_combined_images/1054_predicted.jpg)

---

## 📚 References

- Lin, D. et al., FAIR1M Dataset [IEEE TGRS, 2021](https://huggingface.co/papers/2103.05569)
- Chollet, F. et al., *Keras Library*, 2015
- He, K. et al., *Deep Residual Learning*, CVPR 2016

---

## 📬 Contact

Made with ❤️ by **Rajesh Kumar Jogi**  
📧 [rajeshjogi@email.com](mailto:rajeshjogi@email.com)  
🌐 [LinkedIn](https://www.linkedin.com)
