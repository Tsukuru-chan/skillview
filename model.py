import torch
from torchvision import models, transforms
from PIL import Image

net = models.resnet101(pretrained=True)  # 訓練済みのモデルを読み込み
with open("imagenet_classes.txt") as f:  # ラベルの読み込み
    classes = [line.strip() for line in f.readlines()]

def predict(img):
    # 以下の設定はこちらを参考に設定: https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])

    # モデルへの入力
    img = transform(img)
    x = torch.unsqueeze(img, 0)  # バッチ対応

    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]

# Resnetを転移学習
#http://pchun.work/resnet%E3%82%92fine-tuning%E3%81%97%E3%81%A6%E8%87%AA%E5%88%86%E3%81%8C%E7%94%A8%E6%84%8F%E3%81%97%E3%81%9F%E7%94%BB%E5%83%8F%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%95%E3%81%9B%E3%82%8B/
