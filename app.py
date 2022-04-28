import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

import copy

#loading the model
InceptionV3 = torchvision.models.inception_v3(pretrained=False, progress=False) 

fig = plt.figure()


#background: url("https://wallpapercave.com/wp/wp8099561.jpg");
st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://raw.githubusercontent.com/akha46/images/main/signSpeak.png");
             background-size: cover;
             background-attachment: scroll;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

original_title = '<p style="font-family:Ariel; text-align:center; color:saddlebrown ; font-size:50px; background-color:#FEE1D1;opacity: 0.9;">Sign Speak</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.text("")
st.text("")
st.text("")



class Network(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        #Conv group 1
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size= (3,3)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        
        #Max Pooling
        self.maxpool_1 = nn.MaxPool2d(2)

        #Conv group 2
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= (3,3)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1)
                                   )
        
        #Max Pooling
        self.maxpool_2 = nn.MaxPool2d(2)

        #Conv group 2
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= (3,3)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        
        #Max Pooling
        self.maxpool_3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()       
        self.cls_layer = nn.Sequential(nn.Linear(256, 128),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(128, num_classes)
                                       )  

    def forward(self, x):

      x = self.conv1(x)
      x = self.maxpool_1(x)
      x = self.conv2(x)
      x = self.maxpool_2(x)
      x = self.conv3(x)
      x = self.maxpool_3(x)
      x = self.flatten(x)
      x = self.cls_layer(x)

      return x

def initialize_inceptionv3_model(model, num_classes):
    input_size = 0

    print('Pretrained model has 1000 classes')
    print('changing the classifier layer to predict {} labels...'.format(num_classes))
    model.fc = torch.nn.Linear(2048, num_classes)
    input_size = 400
    return model, input_size



def create_model_input(image):
    # image = image/255 #didn't use normalization here
    image = torch.Tensor(image) 
    image = image.permute(2, 0, 1) 
    
    img_transform = transforms.Compose([transforms.ToPILImage(),
                                        #transforms.Resize((32, 32)),
                                        transforms.ToTensor()])

    image = img_transform(image)
    image = torch.unsqueeze(image, dim=0)
    dataloader = DataLoader(image, batch_size= 1, shuffle=True)

    return dataloader

def label_to_char(pred, mode = None):
    label_to_character = {0:"A",1:"B",2:"M",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J",10:"L",
                          11:"L",12:"M", 13:"N",14:"NOTHING", 15:"O",16:"P",17:"P",18:"R",
                          19:"S",20:"SPACE",21:"T",22:"U",23:"V",24:"W", 25:"X", 26:"Y", 27:"Z"}
    pred = label_to_character[pred]
    if mode == 'video':
        if pred == 'D' or pred == 'P' or pred == 'SPACE':
            pred = 'K'
    return pred
    

def predict(image):
  # Disable grad
  with torch.no_grad():
    
    image = create_model_input(image)

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Loading the saved model
    save_path = './models/ASL_InceptionV3.pth'
    model, _ = initialize_inceptionv3_model(InceptionV3, num_classes = 28)
    model.load_state_dict(copy.deepcopy(torch.load(save_path,device)))

    model.eval()

    for img in image:
        # Generate prediction
        prediction = model(img)  
        # Predicted class value using argmax
        _, preds = torch.max(prediction, 1)
        print(preds)

        return preds
    
    

def main():
   
    selectbox_text = '<p style="font-family:Ariel; text-align:left; color:peru; font-size:30px;">Do you want to upload an image or click a photo</p>'
    st.sidebar.markdown(selectbox_text, unsafe_allow_html=True)
    option = st.sidebar.selectbox("", ('Upload', 'Click Picture'))
    st.sidebar.write('You selected:', option)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    if option == 'Upload':
        upload_image = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Upload an Image</p>'
        st.markdown(upload_image, unsafe_allow_html=True)
        file_up = st.file_uploader("", type="jpg")

        if file_up is not None:    
            image = Image.open(file_up)
            size = (400,400)
            image.thumbnail(size)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            open_cv_image = np.array(image) 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            with st.spinner('Model working....'):
                predictions = predict(open_cv_image)
                character = label_to_char(predictions.item())
                time.sleep(1)
                st.success('Classified')
                st.title(character)

        if file_up is None:
            st.text("")
            st.text("")
            invalid_command = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Please upload an image</p>'
            st.markdown(invalid_command, unsafe_allow_html=True)



    else:
        rendered_frames = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Select number of frames to render:</p>'
        st.markdown(rendered_frames, unsafe_allow_html=True)
        MAX_FRAMES = st.slider("", min_value=30, max_value=90, value=60, step=30)
        run = st.button("Click to render server camera")

        img = None
        if run:
            capture = cv2.VideoCapture(0)
            img_display = st.empty()
            for _ in range(MAX_FRAMES):
                _, img = capture.read()
                img = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
                img_display.image(img, channels='BGR')
            capture.release()
            render_complete = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Render complete</p>'
            st.markdown(render_complete, unsafe_allow_html=True)
            if img is None:
                invalid_command = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Could not read frames</p>'
                st.markdown(invalid_command, unsafe_allow_html=True)
            else:
                with st.spinner('Model working....'):
                    print('predicting the sign...')
                    predictions = predict(img)
                    character = label_to_char(predictions.item(), mode = 'video')
                    time.sleep(1)
                    st.success('Classified')
                    st.title(character)


if __name__ == "__main__":
    main()
