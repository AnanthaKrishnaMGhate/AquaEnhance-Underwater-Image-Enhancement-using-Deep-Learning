🌊 AquaEnhance: Underwater Image Enhancement using Deep Learning
<p align="center"> <img src="assets/output_1.jpg" width="700"> </p>
📌 Overview

AquaEnhance is a deep learning-based underwater image enhancement system designed to restore degraded underwater photographs affected by:

Color distortion
Light attenuation
Low contrast
Scattering and haze
Noise
Blur

The project utilizes a UNet architecture with a ResNet34 encoder trained to reconstruct visually enhanced underwater images while preserving natural colors and fine details. The application is deployed using Streamlit for an interactive user experience.

🚀 Features

✅ Deep Learning-based Underwater Enhancement

✅ UNet + ResNet34 Architecture

✅ Real-time Image Processing

✅ Streamlit Web Interface

✅ PSNR and SSIM Quality Metrics

✅ High Resolution Image Support

✅ Model Checkpoint Loading (.pth)

✅ Image Download Functionality

✅ Before vs After Comparison

🧠 Model Architecture
Input Image
      │
      ▼
ResNet34 Encoder
      │
      ▼
UNet Decoder
      │
      ▼
Enhanced Underwater Image
Model Details
Component	Value
Architecture	UNet
Encoder	ResNet34
Framework	PyTorch
Input Size	256 × 256
Output Channels	3 (RGB)
Activation	Tanh
📂 Repository Structure
AquaEnhance/
│
├── app.py
├── best_model.pth
├── requirements.txt
│
├── assets/
│   ├── input_1.jpg
│   ├── input_2.jpg
│   ├── input_3.jpg
│   ├── input_4.jpg
│   ├── input_5.jpg
│   ├── input_6.jpg
│   │
│   ├── output_1.jpg
│   └── output_2.jpg
│
└── README.md
🖼️ Sample Results
Input Images
Input 1	Input 2	Input 3

	
	
Input 4	Input 5	Input 6

	
	
Enhanced Outputs
Output Example 1

Output Example 2

📊 Evaluation Metrics

The application computes image quality metrics including:

PSNR (Peak Signal-to-Noise Ratio)

Higher values indicate better reconstruction quality.

SSIM (Structural Similarity Index)

Measures structural similarity between images.

SSIM Range: 0 → 1

1 = Perfect Similarity
⚙️ Installation
Clone Repository
git clone https://github.com/yourusername/AquaEnhance.git

cd AquaEnhance
Create Virtual Environment
python -m venv venv
Activate Environment

Windows:

venv\Scripts\activate

Linux / Mac:

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
▶️ Run Application
streamlit run app.py

Open:

http://localhost:8501
🧪 Usage
Launch the Streamlit application.
Upload the trained model (best_model.pth).
Upload an underwater image.
Click process.
View:
Enhanced Image
PSNR
SSIM
Inference Time
Download the enhanced result.
📈 Training

The model was trained using paired underwater image datasets to learn restoration and enhancement mappings.

Training Framework
PyTorch
Albumentations
Segmentation Models PyTorch
CUDA Support
💾 Pretrained Model

Download or use the provided model:

best_model.pth

Place it inside:

Model/best_model.pth

or upload directly through the Streamlit interface.

🛠 Technologies Used
Python
PyTorch
Streamlit
OpenCV
NumPy
Albumentations
Segmentation Models PyTorch
Pillow
📷 Applications
Marine Research
Underwater Robotics
Ocean Exploration
Coral Reef Monitoring
Marine Biodiversity Studies
Underwater Photography Enhancement
