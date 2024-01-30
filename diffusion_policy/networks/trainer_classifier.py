import torch

from diffusion_policy.networks.classifier import ClassifierStageScooping


class TrainerClassifier:
    def __init__(self, in_features, number_of_classes: int = 3):
        self.classifier = ClassifierStageScooping(in_features=in_features, number_of_classes=number_of_classes)
        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), betas=(0.95, 0.999), eps=1.0e-08, lr=0.0001, weight_decay=1.0e-06)

    def train_step(self, x, target_logits):
        self.optimizer.zero_grad()
        loss = self.classifier.compute_loss(x, target_logits)
        loss.backward()
        self.optimizer.step()
        return loss.item()