import json
from PIL import Image
import pytesseract

def extract_annotation_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    width = data['width']
    height = data['height']
    results = data.get('results', [])
    
    annotation_data = []
    for result in results:
        x = result['x']
        y = result['y']
        width = result['width']
        height = result['height']
        annotation_data.append((x, y, width, height))
    
    return annotation_data

def extract_text_from_annotation(image_path, annotation_data):
    image = Image.open(image_path)
    extracted_texts = []
    
    for (x, y, width, height) in annotation_data:
        cropped_image = image.crop((x, y, x+width, y+height))
        text = pytesseract.image_to_string(cropped_image)
        extracted_texts.append(text.strip())
    
    return extracted_texts

# Example usage:
json_file_path = 'image0.json'
image_path = 'image0.jpg'

annotation_data = extract_annotation_data(json_file_path)
extracted_texts = extract_text_from_annotation(image_path, annotation_data)

print("Extracted Texts:")
for idx, text in enumerate(extracted_texts):
    print(f"Annotation {idx+1}: {text}")
