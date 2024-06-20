import os
import streamlit as st
from pdf2image import convert_from_bytes
import cv2
import pytesseract
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from preprocessing.binarization import preprocess_images
from preprocessing.skew import process_skew_correction
from preprocessing.bilateral import process_images

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

# Function to clean the extracted text
def clean_text(input_string):
    # Remove non-alphanumeric characters and extra spaces
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    # Remove extra spaces
    cleaned_string = ' '.join(cleaned_string.split())
    return cleaned_string

# Function to process each image
def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Save result
    cv2.imwrite(output_path, result)

# Function to preprocess each image (binarization, skew correction, and bilateral filtering)
def preprocess_extract_convert(input_binarize):
    # Preprocessing step 1: Binarization
    binarize_output = "preprocessing"
    output_path_binarization = os.path.join(binarize_output,'gray_scale')

    # Clear existing images in the output folder for binarization
    for file in os.listdir(output_path_binarization):
        file_path = os.path.join(output_path_binarization, file)
        os.remove(file_path)
    preprocess_images(input_binarize, output_path_binarization)

    # Preprocessing step 2: Skew correction
    output_path_skew = os.path.join(binarize_output,'skew')
    # Clear existing images in the output folder for skew correction
    for file in os.listdir(output_path_skew):
        file_path = os.path.join(output_path_skew, file)
        os.remove(file_path)
    process_skew_correction(output_path_binarization, output_path_skew)

    # Preprocessing step 3: Bilateral filtering
    output_path_bilateral = os.path.join(binarize_output,'noise')

    # Clear existing images in the output folder for bilateral filtering
    for file in os.listdir(output_path_bilateral):
        file_path = os.path.join(output_path_bilateral, file)
        os.remove(file_path)
    process_images(output_path_skew, output_path_bilateral)

    st.success(f"Preprocessing completed")

    # Extract text from images
    input_folder = output_path_bilateral
    output_file_path = 'extracted.txt'
    with open(output_file_path, 'w') as output_file:
        for filename in os.listdir(input_folder):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                input_path = os.path.join(input_folder, filename)
                extracted_text = extract_text_from_image(input_path)
                output_file.write(extracted_text)
                output_file.write("\n")
    
    # Read data from 'extracted.txt' and clean it
    cleaned_data_list = []
    with open('extracted.txt', 'r') as f:
        for line in f:
            cleaned_line = clean_text(line)
            if cleaned_line:  # Check if line is not empty after cleaning
                cleaned_data_list.append(cleaned_line)

    # Write the cleaned data to 'listresult.txt'
    with open('listresult.txt', 'w') as f:
        for line in cleaned_data_list:
            f.write(line + '\n')

    # Read data from 'listresult.txt' and split into separate columns by spaces
    data_list = []
    with open('listresult.txt', 'r') as f:
        for line in f:
            data_list.append(line.strip().split())

    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(data_list)
     
    # Define the interval
    interval = 11

    # Filter out rows at the specified intervals
    filtered_df = df[(df.index == 0) | (df.index % interval != 0)]

    # Save the filtered DataFrame back to Excel
    filtered_df.to_excel('output.xlsx', index=False, header=False)

    st.write("Excel File Generated")
    
    # Create a button to download the output Excel file
    with open('output.xlsx', 'rb') as f:
        st.download_button(label='Click here to download', data=f, file_name='Result.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    
    st.stop()  # Stop script execution after downloading

def load_data(file_path):
    return pd.read_excel(file_path)

# Function to preprocess data and compute subject-wise mean
def preprocess_and_compute_mean(df):
    # Drop irrelevant columns for subject-wise analysis
    columns_to_drop = ['USN','NAME', 'TOTAL', 'OBTAINED', 'RESULT']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Preprocess the data to remove non-numeric characters
    for col in df.columns:
        # Convert column to string data type
        df[col] = df[col].astype(str)
        # Use .str accessor for string manipulation
        df[col] = df[col].str.extract('(\d+)').astype(float)
    
    # Compute mean marks for each subject
    subject_mean = df.mean()
    return subject_mean

# Function to calculate pass/fail percentage
def pass_fail_percentage(df):
    pass_count = (df['RESULT'] == 'PASS').sum()
    fail_count = (df['RESULT'] == 'FAIL').sum()
    total_students = len(df)
    pass_percentage = (pass_count / total_students) * 100
    fail_percentage = (fail_count / total_students) * 100
    return pass_percentage, fail_percentage

# Function to identify top performers
def identify_top_performers(df, n=9):
    # Sort the DataFrame based on marks obtained in descending order
    df_sorted = df.sort_values(by='OBTAINED', ascending=False)
    
    # Get the top-performing students
    top_performers = df_sorted.head(n)
    return top_performers.tail(n)


# Function to visualize top performers
def visualize_top_performers(top_performers):
    # Plotting the top performers
    plt.figure(figsize=(10, 6))
    plt.bar(top_performers['NAME'], top_performers['OBTAINED'], color='skyblue')
    plt.xlabel('Student Name')
    plt.ylabel('Marks Obtained')
    plt.title('Top Performing Students')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

st.sidebar.title("Academic Performance Analysis")
uploaded_file1 = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file1 is not None:
    bytes_data = uploaded_file1.getvalue()
    df = load_data(uploaded_file1)
    pass_percentage, fail_percentage = pass_fail_percentage(df)

    # Identify top performers
    top_performers = identify_top_performers(df,n=5)

    # Display top performers on the webpage
    st.subheader('Top Performing Students')
    st.write(top_performers)

    # Visualize top performers
    st.subheader('Top Performing Students Visualization')
    visualize_top_performers(top_performers)

    # Preprocess and compute mean
    subject_mean = preprocess_and_compute_mean(df)


    # Display subject-wise analysis
    st.subheader('SUBJECT-WISE ANALYSIS')
    # Customize bar colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(subject_mean)))  # Generate colors from a colormap
    
    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots()
    subject_mean.plot(kind='bar', ax=ax, color=colors)
    ax.set_ylabel('Mean Marks')
    ax.set_title('Subject-wise Mean Marks')

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)

    # Display pass and fail percentages in a unique way
    st.subheader('Pass and Fail Percentages')
    st.markdown(f"Pass Percentage: **{pass_percentage:.2f}%**")
    st.progress(pass_percentage / 100)

    st.markdown(f"Fail Percentage: **{fail_percentage:.2f}%**")
    st.progress(fail_percentage / 100)

    
# Set page title
st.title("Efficient Academic Data Automation")
st.header("PDF to Excel Conversion")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and st.button('Process and Convert'):
    # Convert PDF to images
    images = convert_from_bytes(uploaded_file.read(),poppler_path=r'C:\Users\Admin\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin')

    # Create output folder
    output_folder_input = "inputdata"
    for file in os.listdir(output_folder_input):
        file_path = os.path.join(output_folder_input, file)
        os.remove(file_path)
    os.makedirs(output_folder_input, exist_ok=True)

    # Save each image
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder_input, f"page_{i + 1}.jpg")
        image.save(image_path, 'JPEG')

    st.success("PDF converted to images successfully!")

    # Display the converted images
    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i + 1}", use_column_width=True)

    # Input and output directories
    input_folder = 'inputdata'
    output_folder = 'grid_remove'

    # Create output folder if it doesn't exist
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)

    # Preprocess the image
    preprocess_extract_convert(output_folder)



