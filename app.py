import streamlit as st
import numpy as np
from PIL import Image, ImageChops
import cv2
import io

import transformers
# from transformers import BeitModel, BeitImageProcessor

from sklearn import preprocessing
import numpy as np

import torch


# Load BEiT Model and Processor
mdl = BeitModel.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
img_proc = BeitImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

# Cache for image reading
@st.cache(allow_output_mutation=True)
def read_image_file(file_stream):
    try:
        image = Image.open(io.BytesIO(file_stream.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"An error occurred when reading the image file: {e}")
        return None

# Resize to match
def resize_to_match(image1, image2):
    size_image2 = image2.shape[:2]
    resized_image1 = cv2.resize(image1, (size_image2[1], size_image2[0]))
    return resized_image1, image2

# Correct rotation
def correct_rotation(base_image, image_to_correct):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(base_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image_to_correct, None)

    if descriptors1 is not None and descriptors2 is not None:
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        if not isinstance(matches, list):
            matches = list(matches)

        if len(matches) > 10:
            matches.sort(key=lambda x: x.distance, reverse=False)

            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            if h is not None:
                height, width, channels = base_image.shape
                corrected_image = cv2.warpPerspective(image_to_correct, np.linalg.inv(h), (width, height))
                return corrected_image

    return image_to_correct


# BEiT-related functions
def get_feature(image, model, image_processor):
    inputs = image_processor(image, return_tensors="pt")
        
    #画像(のtensor)をモデルにinput
    with torch.no_grad():
        outputs = model(**inputs)
        
    last_hidden_states = outputs.last_hidden_state
    #special tokenを取り除く
    feature = last_hidden_states[:, 1:, :]
    #モデル入力時にH=224×W=224の画像になっている
    #h = 224
    #w = 224
    #ADE20kの場合
    h = 640
    w = 640
    #データ数、H/16、W/16、hidden_state_sizeにreshape
    feature = torch.reshape(feature, (feature.shape[0], int(h/16), int(w/16), feature.shape[2]))
    
    return feature
    

def patch_upsample(feature):
    #(N, C, H, W)にする
    feature = feature.permute(0, 3, 1, 2) 
    #アップサンプリング
    model_up = nn.Upsample(scale_factor=16, mode='bilinear')
    
    y = model_up(feature)
    
    return y
    

def standardization_and_euclidean_distance(p, q):
    scaler1 = preprocessing.StandardScaler()
    p_scaled = scaler1.fit_transform(p)
    scaler2 = preprocessing.StandardScaler()
    q_scaled = scaler2.fit_transform(q)
    
    distances = np.linalg.norm(p_scaled - q_scaled, axis=0)
    distances = distances.reshape(640, 640)
    
    return distances

#bounding box
def bounding_box(im_euclid, image1, image2):
    #PIL→cv2
    cv2_image = np.array(im_euclid, dtype=np.uint8)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    #gray scale
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    #thresholdで2値化
    th = 20
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
    
    #bounding box描画用の画像
    image1_tmp = image1.copy()
    image2_tmp = image2.copy()
    height1, width1 = image1_tmp.shape[:2]
    height2, width2 = image2_tmp.shape[:2]
    #heatmapを640*640から入力前の大きさにresize
    thresh_r = cv2.resize(thresh, (width1, height1))

    #輪郭を検出、boxを描画    
    counters = cv2.findContours(thresh_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counters = counters[0] if len(counters) == 2 else counters[1]
    for c in counters:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image1_tmp, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.rectangle(image2_tmp, (x, y), (x + w, y + h), (0,255,0), 2)
        
    return image1_tmp, image2_tmp, thresh
    

# Main function for Streamlit app
def main():
    st.title("Image Comparing App")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image-1")
        image1_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'], key="image1")
        if image1_file is not None:
            image1 = read_image_file(image1_file)
            st.image(image1, caption='Uploaded Image 1', use_column_width=True)

    with col2:
        st.header("Upload Image-2")
        image2_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'], key="image2")
        if image2_file is not None:
            image2 = read_image_file(image2_file)
            st.image(image2, caption='Uploaded Image 2', use_column_width=True)

    if st.button('Compare Images'):
        if image1_file is not None and image2_file is not None:
            with st.spinner("Comparing Images..."):
                image1_file.seek(0)
                image2_file.seek(0)
                image1 = read_image_file(image1_file)
                image2 = read_image_file(image2_file)
                if image1 is not None and image2 is not None:
                    resized_image1, resized_image2 = resize_to_match(image1, image2)
                    corrected_image = correct_rotation(resized_image1, resized_image2)
                    corrected_image_pil = Image.fromarray(corrected_image)
                    resized_image2_pil = Image.fromarray(resized_image2)
                    diff = ImageChops.difference(corrected_image_pil, resized_image2_pil).convert('RGB')
                    st.image(diff, caption='Difference', use_column_width=True)
                    st.success("Comparison Completed!")
                else:
                    st.error("Failed to read one or both images.")
        else:
            st.warning("Please upload both images before comparing.")

    if st.button('Compare Images with BEiT'):
        if image1_file is not None and image2_file is not None:
            with st.spinner("Processing with BEiT..."):
                image1_file.seek(0)
                image2_file.seek(0)
                image1 = read_image_file(image1_file)
                image2 = read_image_file(image2_file)
                if image1 is not None and image2 is not None:
                    feature1 = get_feature(image1, mdl, img_proc)
                    feature2 = get_feature(image2, mdl, img_proc)
                    y1 = patch_upsample(feature1)
                    y2 = patch_upsample(feature2)
                    y1 = y1.reshape(y1.shape[0], y1.shape[1], y1.shape[2]*y1.shape[3])
                    y2 = y2.reshape(y2.shape[0], y2.shape[1], y2.shape[2]*y2.shape[3])
                    diff_array_euclid = standardization_and_euclidean_distance(y1[0].detach().numpy(), y2[0].detach().numpy())
                    im_euclid = Image.fromarray(np.array(diff_array_euclid))
                    image1_tmp, image2_tmp, thresh = bounding_box(im_euclid, image1, image2)
                    result_image1 = Image.fromarray(image1_tmp)
                    result_image2 = Image.fromarray(image2_tmp)
                    st.image(result_image1, caption='BEiT Result Image 1', use_column_width=True)
                    st.image(result_image2, caption='BEiT Result Image 2', use_column_width=True)
                    st.success("BEiT Comparison Completed!")
                else:
                    st.error("Failed to read one or both images.")
        else:
            st.warning("Please upload both images before comparing with BEiT.")

if __name__ == "__main__":
    main()
