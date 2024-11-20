# 🧠 Brain Tumor Segmentation using U-Net  

This repository features a **U-Net-based deep learning model** for segmenting brain tumors in MRI images, using the **BraTS'20 dataset**. The model is designed to perform well even with limited data, tackling challenges like class imbalance and preserving spatial details in the images. By using U-Net, it shows how deep learning can be applied effectively to medical image segmentation, specifically for brain tumor detection.

---

## ✨ Features  
- **🔍 Architecture**: U-Net model for semantic segmentation, with skip connections to preserve spatial information.  
- **📊 Dataset**: BraTS’20 dataset - contains ~369 subjects with multimodal MRI scans (T1, T1ce, T2, FLAIR).  
- **⚙️ Loss Functions**: Dice and DiceBCE loss functions - tackle class imbalances and improves segmentation accuracy.  
- **⚡ Optimization**: Adam optimizer.
- **🏆 Performance**: Achieved an average F1-score of 78% on test predictions, outperforming traditional CNNs on the same dataset.  

---

## 🚀 Usage 

1. **Train the Model**  
   ```bash
   python train.py
    ```
2. **Test the Model**
Once trained, evaluate the model with:
  ```bash
  python test.py
  ```

## 📂 Project Structure
data/ - Directory for the BraTS’20 dataset.
models/ - Contains U-Net implementation and saved models.
results/ - Output predictions and performance metrics.
train.py - Script for training the U-Net model.
test.py - Script for testing and visualizing results.

## 🌟 Results
- **🏅 F1-Score:** 78%
- **🔬 Future Work:** Explore ResUNet architecture and experiment with transfer learning to enhance accuracy.

## 🛠️ Dependencies

- Python >= 3.8
- PyTorch >= 1.7
- NumPy
- Matplotlib
- Scikit-learn
