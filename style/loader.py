from PIL import Image
import numpy as np
from torchvision import transforms, models
import cv2

def load_image_or_video(img_path, shape=None):
    in_transform = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        ])
    
    vidObj = cv2.VideoCapture(img_path)
    frames = []
    success = 1
    while success:
        try:
            success, image = vidObj.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        except:
            pass
    return frames


def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
        
    in_transform = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image


def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image
