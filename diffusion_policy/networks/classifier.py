
import torch

# Classifier in PyTorch

class ClassifierStageScooping(torch.nn.Module):
    def __init__(self, in_features, number_of_classes):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, 256)
        self.linear_2 = torch.nn.Linear(256, 256)
        self.linear_3 = torch.nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = torch.nn.functional.selu(self.linear_1(x))
        x = torch.nn.functional.selu(self.linear_2(x))
        x = self.linear_3(x)
        return x

    def compute_loss(self, x, target_logits):
        logits = self.forward(x)
        loss = torch.nn.functional.cross_entropy(logits, target_logits)
        return loss

