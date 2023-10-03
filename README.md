# GhostFaceNet
PyTorch version of [GhostFaceNet](https://github.com/HamadYA/GhostFaceNets/tree/main).

GhostNetV2 code from [Huawei Noah's Ark Lab](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master).

```python
from GhostFaceNetV2 import ghostfacenetv2
import torch

model = ghostfacenetv2(num_classes=3, width=1, dropout=0., args=None)
img = torch.randn(3, 3, 256, 256)
model(img)
```
