import streamlit as st
import PIL.Image as Image
import io
import numpy as np

def display_image_info(image_file):
    if image_file is not None:
        # Reading the image file
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Dimensions: ", image.size)
        st.write("File Size: {:.2f} KB".format(image_file.size / 1024))

def process_image(image, option):
    # Dummy processing function
    # Replace with actual image processing logic
    if option == 'Grayscale':
        image = image.convert('L')
    elif option == 'Flip Horizontal':
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif option == 'Flip Vertical':
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def main():
    st.title("TESTTTTTT-Profield-Image Comparing App")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image-1")
        image1_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], key="image1")
        display_image_info(image1_file)

    with col2:
        st.header("Upload Image-2")
        image2_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], key="image2")
        display_image_info(image2_file)

    processing_option = st.selectbox('Select Image Processing Option', 
                                     ['Grayscale', 'Flip Horizontal', 'Flip Vertical'])

    if st.button('Start Image Processing'):
        if image1_file is not None and image2_file is not None:
            with st.spinner("Processing Images..."):
                image1 = Image.open(image1_file)
                image2 = Image.open(image2_file)

                processed_image1 = process_image(image1, processing_option)
                processed_image2 = process_image(image2, processing_option)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(processed_image1, caption='Processed Image 1', use_column_width=True)
                with col2:
                    st.image(processed_image2, caption='Processed Image 2', use_column_width=True)
            st.success("Processing Completed!")
        else:
            st.warning("Please upload both images before processing.")

if __name__ == "__main__":
    main()
