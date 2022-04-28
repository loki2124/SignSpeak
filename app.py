import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

import copy

fig = plt.figure()


#background: url("https://wallpapercave.com/wp/wp8099561.jpg");
st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/001/437/126/non_2x/sign-language-and-hand-gestures-icon-collection-vector.jpg");
             background-size: cover;
             background-attachment: scroll;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

original_title = '<p style="font-family:Ariel; text-align:center; color:saddlebrown ; font-size:50px; background-color:#FEE1D1;">Sign Speak</p>'
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



def create_model_input(image):
    image = image/255
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1)
    
    img_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor()])

    image = img_transform(image)
    image = torch.unsqueeze(image, dim=0)
    print(image.shape)
    dataloader = DataLoader(image, batch_size= 1, shuffle=True)

    return dataloader
    

def predict(image):
  # Disable grad
  with torch.no_grad():
    
    image = create_model_input(image)

    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Loading the saved model
    save_path = 'MNIST_CNN.pth'
    cnn = Network()
    cnn.load_state_dict(copy.deepcopy(torch.load(save_path,device)))

    cnn.eval()

    for img in image:
        # Generate prediction
        prediction = cnn(img)
        
        # Predicted class value using argmax
        predicted_class = np.argmax(prediction)
        print(predicted_class)

        return predicted_class
    
    

def main():

    dict1={0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"i",9:"k",10:"l",11:"m",12:"n",13:"o",14:"p",15:"q",16:"r",17:"s",18:"t",19:"u",20:"v",21:"w",22:"x",23:"y"}
    
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
        # class_btn = st.button("Classify")
        if file_up is not None:    
            image = Image.open(file_up)
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # if class_btn:
        if file_up is None:
            st.text("")
            st.text("")
            invalid_command = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Invalid command, please upload an image</p>'
            st.markdown(invalid_command, unsafe_allow_html=True)
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = 'test'
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)

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
                img_display.image(img, channels='BGR')
            capture.release()
            render_complete = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Render complete</p>'
            st.markdown(render_complete, unsafe_allow_html=True)
            if img is None:
                invalid_command = '<p style="font-family:Ariel; text-align:left; color:saddlebrown ; font-size:20px; background-color:#FEE1D1;">Invalid command, please upload an image</p>'
                st.markdown(invalid_command, unsafe_allow_html=True)
            else:
                with st.spinner('Model working....'):
                    print('predicting the sign...')
                    predictions = predict(img)
                    time.sleep(1)
                    st.success('Classified')
                    #st.write(predictions)
                    st.title(dict1[predictions.item()])


if __name__ == "__main__":
    main()
