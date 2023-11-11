import torch
import torch.nn as nn

## Network Architecture
class LightWeight_Network(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(LightWeight_Network, self).__init__()

        def conv_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.25))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(16, 2, False), (32, 2, True), (64, 2, True), (128, 2, True)]:
            layers.extend(conv_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        self.model = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class LightWeight_Network_Recog(nn.Module): # With 2 + 1 Conv
    def __init__(self, num_classes=32, pretext_epoch=30):
        super(LightWeight_Network_Recog, self).__init__()
        model = LightWeight_Network()

        def conv_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.25))
            return layers
        
        if(pretext_epoch != 0):
            model.load_state_dict(torch.load("MicroMovement_Weights/epoch-" + str(pretext_epoch) + ".pkl"))
        else:
            print('Without Pre-trained Weights')
            
        self.features = model.model
        self.features = nn.Sequential(*list(self.features.children())[:7])
        self.classifier = nn.Sequential(
            *conv_block(32, 64, 2, True),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out