# 🌊 AquaEnhance

Deep learning-based underwater image enhancement using **UNet + ResNet34** to restore color, visibility, contrast, and fine details in underwater photographs.

---

## ✨ Features

- Underwater image restoration
- UNet with ResNet34 encoder
- Real-time enhancement using Streamlit
- PSNR & SSIM quality metrics
- High-resolution image support
- Download enhanced outputs
- Easy model checkpoint loading (.pth)

---

## 🏗️ Model Architecture

```text
Input Image
     │
     ▼
ResNet34 Encoder
     │
     ▼
UNet Decoder
     │
     ▼
Enhanced Image
```

---

## 📂 Project Structure

```text
AquaEnhance/
│
├── app.py
├── best_model.pth
├── requirements.txt
├── README.md
│
├── Input/
│   ├── input_1.jpg
│   ├── input_2.jpg
│   ├── input_3.jpg
│   ├── input_4.jpg
│   ├── input_5.jpg
│   └── input_6.jpg
│
└── Output/
    ├── output_1.jpg
    └── output_2.jpg
```

---

## 🖼️ Sample Results

### Input Images

| Input 1 | Input 2 | Input 3 |
|----------|----------|----------|
| ![](Input/input_1.jpg) | ![](Input/input_2.jpg) | ![](Input/input_3.jpg) |

| Input 4 | Input 5 | Input 6 |
|----------|----------|----------|
| ![](Input/input_4.jpg) | ![](Input/input_5.jpg) | ![](Input/input_6.jpg) |

### Enhanced Outputs

| Output 1 | Output 2 |
|----------|----------|
| ![](Output/output_1.jpg) | ![](Output/output_2.jpg) |

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/AquaEnhance.git

cd AquaEnhance

pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 💾 Model

Place the trained model file:

```text
best_model.pth
```

in the project root directory or upload it directly from the Streamlit UI.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Streamlit
- OpenCV
- Albumentations
- NumPy
- Pillow
- Segmentation Models PyTorch

---

## 📊 Applications

- Marine Research
- Underwater Robotics
- Coral Reef Monitoring
- Ocean Exploration
- Underwater Photography

---

## 👨‍💻 Author

**Ganesh Prasad**

Data Scientist | AI Engineer

⭐ Star this repository if you found it useful.
