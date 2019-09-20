import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet

import imageio
import matplotlib.pyplot as plt

# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    state_dict = torch.load("checkpoints/model.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def extract_face(input_image):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')

    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    mask = mask[0]
    mask = np.array([mask, mask, mask])
    mask = mask.transpose((1,2,0))

    origin_image = torch_img.squeeze()
    origin_image = origin_image.data.cpu().numpy().transpose((1,2,0))
    output_image = np.where(mask == 0, np.full_like(origin_image, 0), origin_image)

    return output_image

if __name__ == '__main__':
    face_img = extract_face("00025/00025_00058.png")
    imageio.imsave("00025/00025_00058_face.png", face_img)
    plt.imshow(face_img)
    plt.show()