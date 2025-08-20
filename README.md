# U-Net with Self-Attention for MRI Segmentation

This project contains the implementation of a 3D U-Net model with self-attention (SA) designed for MRI brain tumor segmentation. The project was developed as part of my Extended Project Qualification (EPQ) and uses the BraTS 2020 dataset.

---

## Features

- 3D U-Net implementation using PyTorch 
- Self-attention
- Training on BraTS 2020 dataset (multimodal MRI: T1, T1Gd, T2, FLAIR)
- Evaluation with Dice coefficient and Intersection over Union (IoU)
- Streamlit interface for visualization of predicted tumor masks

---

## Setup

1. Clone the repo
```
git clone https://github.com/yourusername/unet-sa-brats2020.git
cd unet-sa-brats2020
```

2. Install dependencies
```
pip install -r requirements.txt
```



## Usage

**Run the interface (GUI)**
```
streamlit run main.py
```

Train the model 
```
python train.py --epochs 100 --batch_size 2
```

## License
This project is for educational purposes only. Please cite appropriately if you use it in academic work.
