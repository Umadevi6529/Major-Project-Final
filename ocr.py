import os
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

# Input and output directories
input_folder = 'preprocessing\\noise'
output_file_path = 'extracted.txt'

# Process each image in the input folder and concatenate the text
with open(output_file_path, 'w') as output_file:
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            extracted_text = extract_text_from_image(input_path)
            output_file.write(extracted_text)
            output_file.write("\n")
