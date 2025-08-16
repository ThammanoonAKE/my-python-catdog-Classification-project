# Cat vs Dog Classification Project 🐱🐶

โปรเจค Deep Learning สำหรับจำแนกรูปภาพแมวและหมาโดยใช้ PyTorch และ Convolutional Neural Network

## 📋 Project Overview

- **Task**: Binary Image Classification (Cat vs Dog)
- **Framework**: PyTorch
- **Model**: Custom CNN Architecture
- **Input**: 224x224 RGB Images
- **Output**: 2 Classes (0=Cat, 1=Dog)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 12.1 (for GPU acceleration, optional)

### Setup Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start
1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook catdog_Classification.ipynb
   ```

2. **Prepare Images:**
   - Place test images (.png, .jpg, .jpeg) in the project directory
   - Run all cells in the notebook

3. **View Results:**
   - Model will classify all images in the folder
   - Results displayed with matplotlib visualization

### Model Loading
The notebook automatically loads the pre-trained model:
```python
model = torch.load("full_model.pth", map_location=torch.device('cpu'))
```

## 📁 Project Structure

```
my-python-catdog-Classification-project/
├── catdog_Classification.ipynb    # Main inference notebook
├── full_model.pth                # Pre-trained PyTorch model
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── CLAUDE.md                     # Development guidance
└── *.png, *.jpg, *.jpeg         # Test images
```

## 🧠 Model Architecture

```python
class MyModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc(x)
        return x
```

**Architecture Details:**
- Convolutional Layer: 3→16 channels, 3x3 kernel
- Max Pooling: 2x2 downsampling
- Fully Connected: 16×112×112 → 2 classes
- Activation: ReLU

## 📊 Model Information

**Pre-trained Model:**
- **Local File**: `full_model.pth` (included in repository)
- **Backup**: [Google Drive Link](https://drive.google.com/file/d/1OKyw8TWGbHPoWToHEuwNB5SVsEIHBBXT/view?usp=sharing)
- **Loading Method**: Complete model (`torch.load()`)

## 🔧 Technical Details

### Image Preprocessing
- **Resize**: 224×224 pixels
- **Normalization**: ImageNet standards
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Color Space**: RGB conversion

### Inference Pipeline
```python
def predict_image(image_path, model):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([...])
    image_tensor = transform(image).unsqueeze(0)
    
    # Device detection and inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    return prediction, image
```

## 🚀 Features

- ✅ Automatic device detection (CUDA/CPU)
- ✅ Batch image processing
- ✅ Real-time visualization
- ✅ RGB image support
- ✅ Error handling for image formats

## 📝 Notes

- Code comments are in Thai language
- Model uses complete serialization (not state_dict)
- Supports both local and cloud environments
- GPU acceleration with CUDA 12.1
- Automatic image discovery in current directory

## 🤝 Contributing

Feel free to submit issues and pull requests for improvements!

## 📄 License

This project is open source. Check the LICENSE file for details.
