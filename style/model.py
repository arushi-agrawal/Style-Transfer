from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
import cv2
from loader import im_convert,load_image,load_image_or_video
from config import *
from helper import gram_matrix, get_features







## YOLO



def function(frames, device, vgg):
    
    final = []
    for content in frames:
        blob = cv2.dnn.blobFromImage(content, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        arr = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    arr.append((confidence,class_id))
        arr.sort(reverse=True)
        obj = classes[arr[0][1]]
        
        style = load_image(style_dict[obj]).to(device)
        content = Image.fromarray(content)
        content = in_transform(content)[:3,:,:].unsqueeze(0).to(device)
        
        # get content and style features only once before training
        content_features = get_features(content, vgg, device)
        style_features = get_features(style, vgg, device)
        
        # calculate the gram matrices for each layer of our style representation
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        
        target = content.clone().requires_grad_(True).to(device)
        
        #show_every = 400
        
        # iteration hyperparameters
        optimizer = optim.Adam([target], lr=0.003)
        steps = 2  # decide how many iterations to update your image (5000)
        
        for ii in range(1, steps+1):
            print(ii)
            # get the features from your target image
            target_features = get_features(target, vgg, device)
            
            # the content loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
            
            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            # then add to it for each layer's gram matrix loss
            for layer in style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                # get the "style" style representation
                style_gram = style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)
        
            # calculate the *total* loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(ii,"done")
        plt.imshow(im_convert(target))
        plt.savefig("static/animated.jpeg")
    final.append(im_convert(target))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## VGG
    vgg = models.vgg19(pretrained=True).features
    print("model downloaded")
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    vgg.to(device)
    frames = load_image_or_video(real_img)
    function(frames,device, vgg)
