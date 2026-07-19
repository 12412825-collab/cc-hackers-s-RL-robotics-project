
Original Model
```text
Image

↓

Your own small CNN

↓

Random Initialization

↓

Training
```

Recommended to change to:

```text
Image

↓

MobileNetV3
(Pretrained)

↓

Extract Image Features

↓

Your own classification layer

↓

Fine-tune
```

That is:

```text
Your own CNN
      ↓
Replace with
Pretrained MobileNet
```

Or:

```text
Your own CNN
      ↓
Replace with
EfficientNet-B0
```

Or:

```text
Your own CNN
      ↓
Replace with
ResNet18
```

---

## Why change it this way?

It mainly has the following advantages:

- **Faster Convergence**: The model already has good initial parameters and doesn\'t need to learn from random weights.
- **Typically Higher Accuracy**: Universal visual features learned by the pretrained model can transfer to the new task.
- **More Friendly to Small Datasets**: Reduces the risk of overfitting.
- **Lower Development Cost**: No need to redesign and validate a new CNN structure.