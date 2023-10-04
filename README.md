# GhostFaceNet
PyTorch version of [GhostFaceNet](https://github.com/HamadYA/GhostFaceNets/tree/main).

GhostNetV2 code from [Huawei Noah's Ark Lab](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master).

Loss function code from [Insight Face](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py).

```python
from GhostFaceNetV2 import ghostfacenetv2
import torch

#return embedding
IMAGE_SIZE = 112
model = ghostfacenetv2(image_size=IMAGE_SIZE, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
display(model(img))

#return classification
IMAGE_SIZE = 112
model = ghostfacenetv2(image_size=IMAGE_SIZE, num_classes=3, width=1, dropout=0., args=None)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
model(img)
```

In order to not use GAP like mentioned in the paper, you need to specify the image size.

# TODO
- [x] Replicate model.
- [ ] Create training code.