import torch
import torch.nn as nn
import torchvision.models as models

class FPN(nn.Module):
    def __init__(self, backbone, out_channels=256):
        super(FPN, self).__init__()
        self.backbone = backbone

        # Define layers for lateral connections
        self.lateral4 = nn.Conv2d(2048, out_channels, 1)
        self.lateral3 = nn.Conv2d(1024, out_channels, 1)
        self.lateral2 = nn.Conv2d(512, out_channels, 1)
        self.lateral1 = nn.Conv2d(256, out_channels, 1)

        # Define layers for top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Define smooth layers to refine the features
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # Extract features from the backbone network
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        c1 = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(c1)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        # Top-down pathway
        p5 = self.lateral4(c5)
        p4 = self.upsample(p5) + self.lateral3(c4)
        p3 = self.upsample(p4) + self.lateral2(c3)
        p2 = self.upsample(p3) + self.lateral1(c2)

        # Smooth the feature maps
        p5 = self.smooth4(p5)
        p4 = self.smooth3(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth1(p2)

        return [p2, p3, p4, p5]

# Example usage:
if __name__ == "__main__":
    # Load a pre-trained ResNet50 model
    resnet = models.resnet50(pretrained=True)

    # Create the FPN using the ResNet backbone
    fpn = FPN(resnet)

    # Example input tensor with batch size of 1 and 3 color channels
    input_tensor = torch.randn(1, 3, 224, 224)

    # Get the output feature maps from the FPN
    output = fpn(input_tensor)

    # Print the shape of each feature map
    for i, o in enumerate(output):
        print(f"Shape of P{i+2}: {o.shape}")
