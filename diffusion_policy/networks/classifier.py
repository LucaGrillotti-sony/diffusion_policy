
import torch
import torch.nn.functional as F

# Classifier in PyTorch

class ClassifierStageScooping(torch.nn.Module):
    def __init__(self, width, height, number_of_classes):
        super().__init__()

        assert width == height

        self.number_of_classes = number_of_classes

        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv_3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool_max = torch.nn.MaxPool2d(2, 2)
        self.pool_avg = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        in_features = (width // 8) * (height // 8) * 16 // 4

        print("IN FEATURES", in_features)

        self.linear_1 = torch.nn.Linear(in_features, 256)
        self.linear_2 = torch.nn.Linear(256, 256)
        self.linear_3 = torch.nn.Linear(256, number_of_classes)


    def forward(self, x):
        x = self.pool_avg(x)
        x = self.pool_max(F.selu(self.conv_1(x)))
        x = self.pool_max(F.selu(self.conv_2(x)))
        x = self.pool_max(F.selu(self.conv_3(x)))
        x = torch.flatten(x, start_dim=-3)  # flattening C x W x H

        x = F.selu(self.linear_1(x))
        x = F.selu(self.linear_2(x))
        x = self.linear_3(x)
        return x

    def compute_loss(self, x, target_logits):
        target_logits = target_logits.ravel()
        logits = self.forward(x)
        logits = logits.view(*target_logits.shape, self.number_of_classes)
        loss = F.cross_entropy(logits, target_logits)
        return loss

    def prediction(self, x, shape_batch=None):
        if shape_batch is None:
            shape_batch = x.shape[:-3]
        logits = self.forward(x)
        logits = logits.view(*shape_batch, self.number_of_classes)
        return torch.argmax(logits, dim=-1)

    def prediction_one_hot(self, x, shape_batch=None):
        return F.one_hot(self.prediction(x, shape_batch), num_classes=self.number_of_classes)

    def accuracy(self, x, target_logits):
        target_logits = target_logits.ravel()
        predicted_logits = self.prediction(x, target_logits.shape)
        return torch.mean(predicted_logits == target_logits, dtype=torch.float32)


def test():
    classifier = ClassifierStageScooping(width=240, height=240, number_of_classes=3)

    num_channels = 3
    h = 240
    w = 240
    random_data = torch.rand(100, num_channels, w, h)
    res = classifier(random_data)
    print(res.shape)
    pred = classifier.prediction_one_hot(random_data)
    print(pred)


if __name__ == '__main__':
    test()
