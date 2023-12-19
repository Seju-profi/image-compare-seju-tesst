import streamlit as st
import numpy as np
from PIL import Image, ImageChops
import cv2
import io

@st.cache(allow_output_mutation=True)
def read_image_file(file_stream):
    try:
        image = Image.open(io.BytesIO(file_stream.read()))
        # Convert the image to RGB if it's not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"An error occurred when reading the image file: {e}")
        return None

def resize_to_match(image1, image2):
    size_image2 = image2.shape[:2]  # Get dimensions of image2
    resized_image1 = cv2.resize(image1, (size_image2[1], size_image2[0]))  # Resize image1
    return resized_image1, image2

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

                    # Convert to PIL images for comparison
                    corrected_image_pil = Image.fromarray(corrected_image)
                    resized_image2_pil = Image.fromarray(resized_image2)

                    # Finding difference
                    diff = ImageChops.difference(corrected_image_pil, resized_image2_pil).convert('RGB')
                    st.image(diff, caption='Difference', use_column_width=True)
                    st.success("Comparison Completed!")
                else:
                    st.error("Failed to read one or both images.")
        else:
            st.warning("Please upload both images before comparing.")

if __name__ == "__main__":
    main()
