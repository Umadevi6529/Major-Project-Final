import cv2
import pytesseract
import xml.etree.ElementTree as ET

# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Return the image
    return image

# Function to extract text from bounding boxes using OCR
def extract_text_from_bounding_boxes(image, xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    extracted_text = []

    # Loop through each object in the XML
    for obj in root.findall('object'):
        # Extract bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Crop the region of interest (ROI) from the image based on the bounding box
        roi = image[ymin:ymax, xmin:xmax]

        # Use pytesseract to extract text from the ROI
        text = pytesseract.image_to_string(roi)

        # Append extracted text to the list
        extracted_text.append(text.strip())

    return extracted_text

# Main function
def main(image_path, xml_file):
    # Load and preprocess image
    image = load_and_preprocess_image(image_path)
    if image is None:
        print("Error: Image file not found.")
        return

    # Extract text from bounding boxes using OCR
    extracted_text = extract_text_from_bounding_boxes(image, xml_file)

    # Print extracted text
    for text in extracted_text:
        print(text)

# Example usage
image_path = 'denoised_rotated_binarized_image0 (1).jpg'
xml_file = 'denoised_rotated_binarized_image0.xml'
main(image_path, xml_file)
