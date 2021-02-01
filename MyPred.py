from net import sphere
import torch
from lfw_eval import extractDeepFeature
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cosface_model():
    model_path = 'cosface_model.pth'
    model = sphere().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_img_feature(img, model):
    img = Image.fromarray(img).convert('RGB')
    feat = extractDeepFeature(img, model, False)
    return feat

def get_distance(f1, f2):
    distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    return distance.item()

if __name__ == "__main__":
    model = load_cosface_model()
    img = np.random.randint(0, 255, [112, 96, 3]).astype(np.uint8)
    img2 = np.random.randint(0, 255, [112, 96, 3]).astype(np.uint8)
    feat = get_img_feature(img, model)
    feat2 = get_img_feature(img2, model)
    dist = get_distance(feat, feat2)
    print(dist)