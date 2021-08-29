import cv2
from torchvision import transforms, models
style_dict = {}
style_dict['dog'] = "images/anime3.png"

real_img = "images/golder-retriever-puppy.jpeg"

net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")
classes = []
with open("YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("YOLO donee")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

## GET WEIGHTS
style_weights = {'conv1_1': 1.,
    'conv2_1': 0.75,
        'conv3_1': 0.2,
            'conv4_1': 0.2,
                'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e9  # beta

in_transform = transforms.Compose([transforms.Resize((256,256)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
