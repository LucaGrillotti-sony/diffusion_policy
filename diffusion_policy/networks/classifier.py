
import torch

# Classifier in PyTorch

class ClassifierStageScooping(torch.nn.Module):
    def __init__(self, in_features, number_of_classes):
        super().__init__()

        self.in_features = in_features
        self.number_of_classes = number_of_classes

        self.linear_1 = torch.nn.Linear(in_features, 256)
        self.linear_2 = torch.nn.Linear(256, 256)
        self.linear_3 = torch.nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = torch.nn.functional.selu(self.linear_1(x))
        x = torch.nn.functional.selu(self.linear_2(x))
        x = self.linear_3(x)
        return x

    def compute_loss(self, x, target_logits):
        target_logits = target_logits.ravel()
        logits = self.forward(x)
        logits = logits.view(*target_logits.shape, self.number_of_classes)
        loss = torch.nn.functional.cross_entropy(logits, target_logits)
        return loss

    def prediction(self, x, shape_batch):
        logits = self.forward(x)
        logits = logits.view(*shape_batch, self.number_of_classes)
        return torch.argmax(logits, dim=-1)

    def accuracy(self, x, target_logits):
        target_logits = target_logits.ravel()
        predicted_logits = self.prediction(x, target_logits.shape)
        return torch.mean(predicted_logits == target_logits, dtype=torch.float32)



