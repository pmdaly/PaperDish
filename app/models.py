import json
import torch
import torch.nn.functional as F
from torchvision import transforms


class ImageNet:

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor()
        ])
    labels = {int(k): v for k, v in
              json.load(open('imagenet_class_index.json')).items()}


class AlexNet:

    def __init__(self, transform=ImageNet.transform, labels=ImageNet.labels,
                       pretrained=True):
        self.model = self._load_model(pretrained)
        self.transform = transform
        self.labels = labels
        self.pretrained = pretrained

    def _load_model(self, pretrained):
        if pretrained:
            return torch.hub.load('pytorch/vision:v0.6.0',
                                  'alexnet', pretrained=True)
        else:
            # load personal implementation
            return

    def __call__(self, image, topk=5):
        image_tensor = self.transform(image)
        image_batch = image_tensor.unsqueeze(0)
        with torch.no_grad():
            preds = self.model(image_batch)
        probas, idxs  = F.softmax(preds[0], dim=0).topk(topk)
        return probas, [self.labels[idx.item()][1] for idx in idxs]

    def eval(self):
        self.model.eval()


def topk_to_rank_string(probas, labels):
    out = ''
    for proba, label in zip(probas, labels):
        proba = round(proba.item()*100, 3)
        label = label.replace('_', ' ').title()
        out += f'{label}: {proba}%<br/>'
    return out
