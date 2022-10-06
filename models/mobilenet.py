from torch import nn
import torchvision.models as tvmodels

def get_mobilenet(n_dim):
    model = tvmodels.mobilenet_v2(weights=tvmodels.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_dim)

    return model