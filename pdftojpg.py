import os
from pdf2image import convert_from_path

# PDF file path
pdf_path = 'C:\\Users\\Admin\\Major Project Final\\6th Semester Provisional Result Sheet.pdf'

# Output folder path
output_folder = 'images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path,poppler_path=r'C:\Users\Admin\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin')

# Save each image
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
    image.save(image_path, 'JPEG')
    print(f"Image saved: {image_path}")
