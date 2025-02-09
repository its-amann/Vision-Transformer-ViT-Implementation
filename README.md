# Vision Transformer (ViT) Implementation 🚀

<div align="center">

<img src="_asserts\vit_architecture_image.png" width="800px" alt="Vision Transformer Architecture"/>

<br>

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/models)

</div>

<div align="center">
<h3>🎯 A PyTorch Implementation of the Vision Transformer (ViT) Paper</h3>
<h4><i>"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"</i></h4>
</div>

---

<div align="center">

### 🌟 Key Features

| Feature | Description |
|---------|------------|
| 🔨 **Complete ViT** | Full implementation of Vision Transformer |
| 📊 **Custom Pipeline** | Flexible training and data processing |
| 🎯 **Fine-tuning** | Support for transfer learning |
| 📈 **Visualization** | Comprehensive model insights |
| 🤗 **HuggingFace** | Pre-trained model integration |
| 🖼️ **Custom Datasets** | Support for custom image data |

</div>

---

## 📚 Contents

<details>
<summary>Click to expand</summary>

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Implementation Details](#-implementation-details)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Results](#-results)
- [Pre-trained Models](#-pre-trained-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)

</details>

## 🎯 Dataset: FoodVision Mini

<div align="center">

### 🍽️ Three-Class Food Classification

<table>
<tr>
<td align="center">🍕 Pizza</td>
<td align="center">🥩 Steak</td>
<td align="center">🍣 Sushi</td>
</tr>
<tr>
<td><img src="_asserts/visualize the single image.png" width="200px"/></td>
<td><img src="_asserts/predicting using our model.png" width="200px"/></td>
<td><img src="_asserts/seeing the single image to turn it into the patch.png" width="200px"/></td>
</tr>
</table>

</div>

## 💫 Model Architecture

<div align="center">

### 🔄 Vision Transformer Pipeline

<img src="_asserts/visual diagram of our vison transformer.png" width="800px"/>

### 🧩 Patch Embedding Process

<table>
<tr>
<td><img src="_asserts/converted our images into the patch viz.png" width="250px"/></td>
<td><img src="_asserts/seeing the top row only of the patch.png" width="250px"/></td>
<td><img src="_asserts/seeing all the things together,image , feature map and flattened images.png" width="250px"/></td>
</tr>
<tr>
<td align="center"><i>Image Patches</i></td>
<td align="center"><i>Feature Maps</i></td>
<td align="center"><i>Embeddings</i></td>
</tr>
</table>

### 🔍 Transformer Encoder

<img src="_asserts/summary of the encoder block that we have made.png" width="800px"/>

</div>

## 📊 Training Results

<div align="center">

### 📈 Performance Metrics

<table>
<tr>
<td>
<img src="_asserts/plotting of the loss and accuracy of the trained model.png" width="400px"/>
<br>
<i>Training Progress</i>
</td>
<td>
<img src="_asserts/training the vit.png" width="400px"/>
<br>
<i>Model Metrics</i>
</td>
</tr>
</table>

### 🎯 Model Performance

| Metric | Value |
|--------|--------|
| Training Accuracy | 98% |
| Validation Accuracy | 91% |
| Model Size | 327MB |
| Training Time | ~30 min (CUDA) |

</div>

## 🚀 Pre-trained Model Results

<div align="center">

### 🔄 Transfer Learning Performance

<table>
<tr>
<td>
<img src="_asserts/results of the pretrained model.png" width="400px"/>
<br>
<i>Pre-trained Model Results</i>
</td>
<td>
<img src="_asserts/plots of the pretrained model.png" width="400px"/>
<br>
<i>Performance Curves</i>
</td>
</tr>
</table>

</div>

## 🌟 Overview

This project implements the Vision Transformer (ViT) architecture from scratch using PyTorch, following the original paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). The implementation includes both training from scratch and utilizing pre-trained models from HuggingFace.

### Key Features:
- 🔨 Complete implementation of ViT architecture
- 📊 Custom training pipeline
- 🎯 Fine-tuning capabilities
- 📈 Performance visualization
- 🤗 Integration with HuggingFace models
- 🖼️ Support for custom image datasets

## 🎯 Dataset Description

This implementation uses a focused subset of food images consisting of three specific classes:
1. 🍕 Pizza
2. 🥩 Steak
3. 🍣 Sushi

### Dataset Details:
- **Source**: Custom FoodVision Mini dataset
- **Classes**: 3 (Pizza, Steak, Sushi)
- **Image Size**: 224x224 pixels
- **Format**: RGB images
- **Split Ratio**: 80% training, 20% testing

### Sample Images:
<div align="center">

![Single Image Visualization](_asserts/visualize%20the%20single%20image.png)
*Example of a single image from our dataset*

</div>

### Data Distribution:
```
Dataset Structure:
└── data/
    ├── train/
    │   ├── pizza/
    │   ├── steak/
    │   └── sushi/
    └── test/
        ├── pizza/
        ├── steak/
        └── sushi/
```

## 🗂 Project Structure

```mermaid
graph TD
    A[Project Root] --> B[src/]
    A --> C[notebooks/]
    A --> D[_assets/]
    A --> E[models/]
    A --> F[data/]
    
    B --> G[helper_functions.py]
    B --> H[engine.py]
    B --> I[data_setup.py]
    
    C --> J[ViT_Implementation.ipynb]
    
    D --> K[Images and Diagrams]
    
    E --> L[Saved Models]
    
    F --> M[Training Data]
    F --> N[Test Data]
```

## 💡 Implementation Details

### 1. Data Processing and Augmentation
The implementation processes the food images through several stages:

```python
# Data augmentation and preprocessing pipeline
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### Visualization of Data Processing Steps:
<div align="center">

![Patch Creation Process](_asserts/seeing%20the%20single%20image%20to%20turn%20it%20into%20the%20patch.png)
*Process of converting an image into patches*

![Top Row Patches](_asserts/seeing%20the%20top%20row%20only%20of%20the%20patch.png)
*Visualization of patches from the top row*

![Complete Patch Visualization](_asserts/converted%20our%20images%20into%20the%20patch%20viz.png)
*Complete image broken down into patches*

</div>

### 2. Patch Embedding Process
The image patching process:

<div align="center">

![Patch Embedding Example](_asserts/example%20working%20of%20the%20aassertion%20in%20the%20patchEmbedding%20class.png)
*Patch Embedding class assertion and functionality*

![Flattened Features](_asserts/vizlualizing%20the%20single%20flatten%20layer.png)
*Visualization of flattened feature layer*

![Combined Visualization](_asserts/seeing%20all%20the%20things%20together,%20image%20,%20feature%20map%20and%20flattened%20images.png)
*Combined view of image, feature maps, and flattened representations*

</div>

Key implementation details:
- Input image size: 224x224 pixels
- Patch size: 16x16 pixels
- Number of patches: 196 (14x14)
- Embedding dimension: 768

### 3. Model Architecture Visualization

```mermaid
graph TD
    A[Input Image 224x224] --> B[Split into 16x16 Patches]
    B --> C[Patch + Position Embeddings]
    C --> D[Transformer Encoder Blocks]
    D --> E[MLP Head]
    E --> F[3 Class Prediction]

    subgraph "Patch Details"
        G[16x16 Patches] --> H[196 Total Patches]
        H --> I[768D Embeddings]
    end
```

### 4. Training Results

#### Model Performance:
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~91%
- **Model Size**: 327MB
- **Training Time**: ~30 minutes on CUDA GPU

#### Loss and Accuracy Curves:
[PLACEHOLDER FOR TRAINING CURVES]

#### Feature Map Visualizations:
[PLACEHOLDER FOR FEATURE MAPS]

### 5. Inference Examples

Sample predictions on test images:
[PLACEHOLDER FOR INFERENCE EXAMPLES]

## 🚀 Training Process

The training implementation follows a comprehensive approach:

```python
# Training configuration
config = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 10,
    "embedding_dim": 768,
    "num_heads": 12,
    "patch_size": 16,
    "num_classes": 3,  # Pizza, Steak, Sushi
    "dropout": 0.1
}
```

### Training Stages:
1. Data Preparation
   - Loading and preprocessing images
   - Splitting into train/test sets
   - Creating data loaders

2. Model Setup
   - Initializing ViT architecture
   - Setting up optimizer and loss function
   - Configuring learning rate scheduler

3. Training Loop
   - Forward pass
   - Loss calculation
   - Backpropagation
   - Parameter updates
   - Metrics tracking

4. Validation
   - Model evaluation
   - Performance metrics calculation
   - Best model checkpointing

## 🏗 Model Architecture

The complete ViT architecture:

```mermaid
graph TD
    A[Input Image] --> B[Patch Embedding]
    B --> C[Position Embedding]
    C --> D[Transformer Encoder x12]
    D --> E[MLP Head]
    E --> F[Classification]
    
    subgraph "Transformer Encoder Block"
    G[Multi-Head Attention] --> H[Add & Norm]
    H --> I[Feed Forward]
    I --> J[Add & Norm]
    end
```

### Position and Class Token Embedding:
<div align="center">

![Class Token Addition](_asserts/added%20the%20class%20token%20in%20the%20number%20of%20patches.png)
*Adding class token to patch embeddings*

![Expanded Class Embedding](_asserts/expanded%20the%20class%20embedding%20to%20the%20batch%20sizebatch%20size.png)
*Expanding class embedding to match batch size*

![Position Embedding](_asserts/patch%20plus%20positional%20embedding%20added%20ready%20to%20enter%20into%20the%20block.png)
*Adding positional embeddings to patches*

</div>

### Transformer Architecture:
<div align="center">

![Visual Diagram](_asserts/visual%20diagram%20of%20our%20vison%20transformer.png)
*Complete visual diagram of our Vision Transformer*

![MSA to MLP](_asserts/values%20changed%20from%20msa%20to%20mlp%20block.png)
*Value transformations through MSA and MLP blocks*

![Encoder Summary](_asserts/summary%20of%20the%20encoder%20block%20that%20we%20have%20made.png)
*Summary of our custom encoder block*

![PyTorch Transformer](_asserts/made%20the%20transformer%20encoder%20using%20the%20pytorch%20inbluid%20transformer%20layer.png)
*Implementation using PyTorch's transformer layers*

</div>

### Model Summary and Architecture:
<div align="center">

![Custom ViT Summary](_asserts/over%20custom%20vit%20summary.png)
*Summary of our custom ViT model*

</div>

### Training Process:
<div align="center">

![Training Start](_asserts/training%20the%20vit%20has%20been%20started%20.png)
*Beginning of model training*

![Training Progress](_asserts/training%20the%20vit.png)
*Training progress and metrics*

</div>

### Training Results and Visualization:
<div align="center">

![Loss and Accuracy Plots](_asserts/plotting%20of%20the%20loss%20and%20accuracy%20of%20the%20trained%20model.png)
*Training and validation metrics over time*

</div>

### Pre-trained Model Integration:
<div align="center">

![Pretrained Model Download](_asserts/pretrainied%20model%20downloaded.png)
*Loading pretrained weights*

![Pretrained Architecture](_asserts/pretrainied%20model%20architecture%20visualization.png)
*Architecture of the pretrained model*

![Pretrained Results](_asserts/results%20of%20the%20pretrained%20model.png)
*Performance results with pretrained model*

![Pretrained Plots](_asserts/plots%20of%20the%20pretrained%20model.png)
*Performance plots for pretrained model*

</div>

### Model Deployment:
<div align="center">

![Model Saving](_asserts/saving%20the%20trained%20model.png)
*Saving the trained model to disk*

![Model Predictions](_asserts/predicting%20using%20our%20model.png)
*Making predictions on new images*

</div>

### Feature Visualization:
<div align="center">

![Single Image](_asserts/visualize%20the%20single%20image.png)
*Single image input visualization*

![Patch Creation](_asserts/seeing%20the%20single%20image%20to%20turn%20it%20into%20the%20patch.png)
*Process of creating patches from image*

![Top Row Patches](_asserts/seeing%20the%20top%20row%20only%20of%20the%20patch.png)
*Visualization of top row patches*

![Complete Patches](_asserts/converted%20our%20images%20into%20the%20patch%20viz.png)
*Complete patch-wise decomposition*




</div>

## 🤗 Using Pre-trained Model

Integration with HuggingFace pre-trained models:
1. Model loading
2. Weight freezing
3. Fine-tuning process
4. Inference pipeline

```python
# Example of loading pre-trained model
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
```

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vit-implementation.git

# Install dependencies
pip install -r requirements.txt
```

Required dependencies:
- PyTorch >= 2.0
- torchvision
- numpy
- matplotlib
- requests
- tqdm

## 📝 Usage

```python
# Example usage
from vit_model import ViT

# Create model instance
model = ViT(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embedding_dim=768,
    num_heads=12,
    num_layers=12,
    hidden_dim=3072,
    dropout=0.1
)

# Train model
train_model(model, train_loader, val_loader, num_epochs=10)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## 🙏 Thank You 💫

<div align="center">

```ascii
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣀⢠⡤⠤⠖⠒⠒⠒⠲⣆⠀⠀⠀⠀⣾⠋⠉⠉⠛⢷⠀⣴⠖⠒⠤⣄⠀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⠤⠶⢺⣾⣏⠁⠀⠀⣧⣼⣇⣀⠀⠀⠀⡀⠀⠘⡆⠀⠀⢰⣏⠀⠀⠀⠀⠘⣿⡟⠀⠀⢠⢃⣼⡏⠉⠙⢳⡆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣀⡤⠴⠒⠋⠙⣇⣿⠀⠀⠀⣿⣿⠀⠀⠀⢸⣿⣿⣿⠃⠀⢰⣿⡀⠀⠹⡄⠀⢸⣿⠀⠀⠀⠀⠀⢹⡇⠀⠀⢸⡿⣽⠀⠀⠀⡜⠀⣀⡤⠖⠓⠢⢤⣀⠀
⣠⡴⠒⠉⠁⠀⠀⠀⠀⠀⠸⣿⡇⠀⠀⠘⠛⠃⠀⠀⠈⡟⠉⣿⠀⠀⠘⠛⠃⠀⠀⢷⠀⢸⣿⠀⠀⢠⡀⠀⠀⠀⠀⠀⣿⢧⡇⠀⠀⠸⠗⠚⠁⠀⠀⠀⣀⣠⣾⠃
⣿⡇⠀⠀⠀⠀⠀⠀⣶⣶⣿⢿⢹⠀⠀⠀⢀⣀⠀⠀⠀⢳⠀⣿⠀⠀⢀⣀⣤⠀⠀⠘⣇⢸⡏⠀⠀⢸⣧⠀⠀⠀⠀⢸⣿⡿⠀⠀⢀⠀⠀⠀⢀⣤⣶⣿⠿⠛⠁⠀
⢧⣹⣶⣾⣿⡄⠀⠀⠸⡟⠋⠘⡜⡆⠀⠀⢻⣿⡇⠀⠀⢸⡀⣿⠀⠀⢸⣿⡿⡇⠀⠀⢸⣿⡇⠀⠀⢸⡿⡆⠀⠀⠀⣾⣿⠃⠀⠀⣾⡇⠀⠀⠈⡟⠉⠀⠀⠀⠀⠀
⠘⣿⡿⠿⢿⣧⠀⠀⠀⢳⡀⠀⣇⢱⠀⠀⠈⣿⣷⠀⣀⣸⣷⣿⣤⣤⣼⠋⣇⣹⣶⣶⣾⣿⡿⢲⣶⣾⡇⣿⣤⣀⣀⣿⡏⠀⠀⣼⡏⢧⠀⠀⠀⣇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠸⡞⣇⠀⠀⠀⢧⠀⢸⣈⣷⣶⣶⣿⣿⣿⣿⣿⣿⣿⣽⣿⡏⢀⡼⠟⠛⠻⢿⡿⠿⠿⣿⣁⣿⣿⣿⣿⣿⣿⣿⣶⣴⢿⠁⢸⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢹⣼⣦⣤⣶⣿⠁⣀⣿⠿⠿⣿⣫⣿⠉⠁⠀⠀⠀⡏⠀⣴⠏⠀⠀⠀⠀⠀⠹⣆⠀⢠⣿⠀⠀⠀⢈⠟⢻⡿⠿⣅⣘⡆⣸⣇⠀⠀⢸⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠻⠿⠿⠛⠃⢠⣿⣷⣄⠀⠈⠙⠋⠀⠀⠀⠀⣸⢁⡾⠁⠀⠀⣠⣤⡀⠀⠀⠸⣤⡞⡇⠀⠀⠀⢸⣰⣿⠃⠀⠀⢹⣿⣿⣿⣿⣦⣼⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⢿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⣿⣾⠇⠀⠀⣸⣿⣿⢿⠀⠀⠀⣿⢁⡇⠀⠀⢀⣿⣿⡏⠀⠀⠀⡼⠀⢙⣿⠛⠻⣏⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣷⠀⠀⠀⠀⢸⡿⡿⠀⠀⠀⡏⢹⠟⡟⠀⠀⠀⡿⢸⠀⠀⠀⢸⣿⡿⠀⠀⠀⢠⠇⡰⢋⡏⠀⠀⠀⢙⡆⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⡿⡿⠀⠀⠀⠀⣸⡇⡇⠀⠀⠀⠻⠾⠞⠁⠀⠀⢀⡇⡏⠀⠀⠀⢸⣿⠃⠀⠀⠀⡼⣰⠃⡞⠀⠀⠀⠀⡾⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇⡇⠀⠀⠀⠀⣿⣇⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣃⡇⠀⠀⠀⠀⠀⠀⠀⠀⣼⣷⠃⣼⡀⠀⠀⢀⡞⠁⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠃⠀⠀⠀⢀⡇⢿⣿⣧⣀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⣽⣿⣷⣤⡞⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣼⣤⣶⣶⣶⡿⠁⠈⢿⣿⣿⣿⣿⣿⣿⣿⠿⠃⢸⣿⣿⣷⣤⣄⣀⣀⣤⣾⣏⣤⡟⠁⠀⠈⠻⡍⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠿⠿⠿⠟⠛⠁⠀⠀⠀⠉⠛⠛⠛⠛⠉⠁⠀⠀⠀⠙⠿⢿⣿⣿⡿⠿⠋⢀⣿⣿⣧⡀⠀⠀⣠⡇⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                                                         
```

Thank you for exploring this Vision Transformer implementation! If you find this project helpful, feel free to use it in your own Vision Transformer tasks. Your interest and support are greatly appreciated! 🌟