# Vision Transformer (ViT) for MNIST - From Scratch

A complete from-scratch implementation of the Vision Transformer (ViT) architecture applied to the MNIST dataset. This project implements every component without relying on pre-built transformer libraries, providing a clear understanding of how ViT works internally.

![Vision Transformer Architecture]([https://raw.githubusercontent.com/yourusername/yourrepo/main/architecture.png](https://www.google.com/imgres?q=vit%20transformer%20research%20paper&imgurl=https%3A%2F%2Fmiro.medium.com%2F1*AoE_mecs9_prJ2mIc9TDqQ.png&imgrefurl=https%3A%2F%2Fai.plainenglish.io%2Fvision-transformers-explained-from-paper-to-pytorch-implementation-8ab20957f0b0&docid=pUSuJLx25oBQlM&tbnid=fg-VNxCQhLJ-GM&vet=12ahUKEwiZ9eSL_JSRAxXFVEEAHddeBHYQM3oECBoQAA..i&w=930&h=485&hcb=2&ved=2ahUKEwiZ9eSL_JSRAxXFVEEAHddeBHYQM3oECBoQAA))
*Architecture diagram showing the flow from image patches through the transformer encoder to classification*

## Overview

This notebook implements a Vision Transformer model that:
- Splits 28×28 MNIST images into 4×4 patches (49 patches per image)
- Projects patches into 128-dimensional embeddings using convolutional layers
- Adds a learnable CLS token for classification
- Processes patches through 10 transformer encoder layers
- Classifies digits (0-9) using an MLP head

## Architecture Components

All components are implemented from scratch:

### 1. PatchEmbedding
- Uses Conv2d with `kernel_size=stride=patch_size` to extract and embed patches
- Converts 28×28 images into 49 patch tokens
- Each patch is embedded into a 128-dimensional vector

### 2. LayerNorm
- Custom layer normalization implementation
- Normalizes across the embedding dimension
- Includes learnable weight and bias parameters

### 3. Gelu Activation
- Custom GELU (Gaussian Error Linear Unit) implementation
- Uses tanh approximation for smooth non-linearity

### 4. MultiHeadAttention
- Implements scaled dot-product attention
- 8 attention heads by default
- Separate linear projections for Q, K, V
- Includes dropout for regularization

### 5. MLP (Feed-Forward Network)
- Two-layer network with 4×embed_dim expansion
- GELU activation in between
- Projects back to embed_dim

### 6. TransformerEncoder
- Pre-normalization architecture (LayerNorm before attention/MLP)
- Residual connections around both attention and MLP blocks
- Stacks multiple blocks for depth

### 7. ViT Model
- CLS token prepended to patch embeddings
- 10 transformer encoder blocks
- MLP head for final classification

## Hyperparameters

```python
batch_size = 32
patch_size = 4          # 4×4 patches
embed_dim = 128         # Embedding dimension
in_channel = 1          # Grayscale images
n_blocks = 10           # Number of transformer layers
n_classes = 10          # MNIST digits (0-9)
num_heads = 8           # Multi-head attention heads
eps = 1e-5              # LayerNorm epsilon
```

## Trainer Class

Custom training loop with:
- **Early stopping**: Monitors validation loss with configurable patience
- **Learning rate scheduling**: ReduceLROnPlateau scheduler
- **Model checkpointing**: Saves best model based on validation loss
- **Metrics tracking**: Logs training/validation loss and accuracy per epoch

### Training Features
- Automatic device selection (CUDA/CPU)
- Batch-wise training with gradient accumulation
- Accuracy computation using sklearn's accuracy_score
- Progress tracking for each epoch

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0 (optional, for visualization)
```

## Usage

### In Jupyter Notebook

```python
# Load MNIST data
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = ViT(n_blocks=10, n_classes=10, patch_size=4, 
            embed_dim=128, in_channel=1)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3)

# Create trainer
trainer = Trainer(
    optimizer=optimizer,
    model=model,
    loss=loss_fn,
    train_dl=train_dl,
    val_dl=val_dl,
    lr_scheduler=lr_scheduler,
    patience=5
)

# Train model
history = trainer.fit(n_epochs=50)
```

### Saved Model

The best model is automatically saved as `best_vit.pth` during training.

## Implementation Highlights

### Custom Components
- **No torch.nn.MultiheadAttention**: Built from scratch with Q, K, V projections
- **No torch.nn.LayerNorm**: Custom implementation with learnable parameters
- **No torch.nn.GELU**: Mathematical implementation using tanh approximation

### Architecture Decisions
- **Pre-normalization**: LayerNorm applied before attention and MLP (more stable training)
- **No positional embeddings**: Relies on learned position information through CLS token and attention
- **Conv2d for patching**: Efficient patch extraction using convolution with stride

### Training Features
- **Early stopping**: Prevents overfitting by monitoring validation loss
- **LR scheduling**: Reduces learning rate on plateau for better convergence
- **Best model saving**: Keeps the model with lowest validation loss

## Project Structure

```
vit-mnist-notebook.ipynb
├── Imports and Setup
├── Hyperparameters
├── Model Components
│   ├── PatchEmbedding
│   ├── LayerNorm
│   ├── Gelu
│   ├── MultiHeadAttention
│   ├── MLP
│   ├── TransformerEncoder
│   └── ViT
├── Trainer Class
├── Data Loading
├── Training Loop
├── Evaluation
└── Visualization (optional)
```

## Expected Results

With the default hyperparameters:
- **Training accuracy**: 98-99%
- **Validation accuracy**: 97-98%
- **Training time**: ~10-20 minutes on GPU (depends on epochs and early stopping)

## Key Insights

This implementation demonstrates:
1. **Patch-based processing**: How images are tokenized for transformers
2. **Self-attention for vision**: Spatial relationships learned through attention
3. **CLS token classification**: Using a learnable token for aggregation
4. **Residual connections**: Crucial for training deep transformer networks
5. **Pre-normalization**: More stable than post-normalization for deep networks

## Advantages of This Implementation

- ✅ **Educational**: Every component is visible and understandable
- ✅ **Customizable**: Easy to modify any part of the architecture
- ✅ **Self-contained**: No external transformer dependencies
- ✅ **Well-commented**: Clear variable names and structure

## Potential Improvements

- Add positional embeddings for better spatial awareness
- Implement attention visualization
- Add data augmentation for better generalization
- Experiment with different patch sizes (7×7, 14×14)
- Try different attention mechanisms (e.g., linear attention)

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Original ViT paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Dataset information

## License

MIT License - Feel free to use this code for learning and experimentation.

## Acknowledgments

- Original ViT architecture by Dosovitskiy et al. (Google Research)
- MNIST dataset by Yann LeCun et al.
- Inspired by the goal of understanding transformers from first principles

---
