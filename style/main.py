from flask import *
import os
import pandas as pd
from forms import *
from model import function
import torch
from torchvision import models
from loader import load_image_or_video

import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
## VGG
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
    
vgg.to(device)

app = Flask(__name__)


#page1
@app.route('/')
def upload():
    return render_template("index.html")

@app.route('/success',methods = ['POST'])
def success():
    
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("static", f.filename))
        print(f.filename)
        
        frames = load_image_or_video("static/"+f.filename)
        function(frames,device, vgg)
        
        return render_template("success.html", name = f.filename, BASE_URL="", anim_img = "animated.jpeg" )




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)






