import streamlit as st
import PIL.Image as Image
import PIL.ImageChops as ImageChops
import cv2
import numpy as np
import io

def resize_to_match(image1, image2):
    size_image2 = image2.shape[:2]  # Get dimensions of image2
    resized_image1 = cv2.resize(image1, (size_image2[1], size_image2[0]))  # Resize image1
    return resized_image1, image2

def read_image_file(file):
    try:
        # Read image file and convert to RGB format
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"An error occurred when reading the image file: {e}")
        return None


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

def preprocess_images_for_comparison(image1, image2):
    image1, image2 = resize_to_match(image1, image2)
    image1 = correct_rotation(image2, image1)
    return image1, image2

def read_image_file(file):
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        return np.array(image)
    except Exception as e:
        st.error(f"An error occurred when reading the image file: {e}")
        return None

def display_image_info(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Dimensions: ", image.size)
        st.write("File Size: {:.2f} KB".format(image_file.size / 1024))

def main():
    st.title("Image Comparing App")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image-1")
        image1_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], key="image1")
        display_image_info(image1_file)

    with col2:
        st.header("Upload Image-2")
        image2_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], key="image2")
        display_image_info(image2_file)

    if st.button('Compare Images'):
        if image1_file is not None and image2_file is not None:
            with st.spinner("Comparing Images..."):
                image1 = read_image_file(image1_file)
                image2 = read_image_file(image2_file)

                if image1 is not None and image2 is not None:
                    preprocessed_img1, preprocessed_img2 = preprocess_images_for_comparison(image1, image2)

                    # Convert to PIL images for comparison
                    preprocessed_img1 = Image.fromarray(preprocessed_img1)
                    preprocessed_img2 = Image.fromarray(preprocessed_img2)

                    # Finding difference
                    diff = ImageChops.difference(preprocessed_img1, preprocessed_img2)

                    # Display the result
                    st.image(diff, caption='Difference', use_column_width=True)

                    st.success("Comparison Completed!")
                else:
                    st.error("Failed to read one or both images.")
        else:
            st.warning("Please upload both images before comparing.")

if __name__ == "__main__":
    main()
