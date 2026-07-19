 Backbone

A Backbone is a neural network responsible for **extracting image features**.

For example:

```text
Image
 ↓
Backbone
 ↓
Feature Vector
 ↓
Classifier
```

Common Backbones:

- ResNet18
- ResNet50
- MobileNetV3
- EfficientNet
- ConvNeXt
- ViT (Vision Transformer)

It can be understood as:

> A Backbone is like a human eye, responsible for turning an image into features that the machine can understand.

---

### (3) Lightweight Backbone

Lightweight means fewer parameters and faster computation.

For example:

| Model | Parameter Count |
|---|---|
| ResNet50 | 25 Million |
| MobileNetV3 | Around 5 Million |
| EfficientNet-B0 | 5.3 Million |
| ShuffleNet | Around 1 Million |

Lightweight models are suitable for:

- Small datasets
- Weak GPUs
- Real-time inference
- Graduation projects