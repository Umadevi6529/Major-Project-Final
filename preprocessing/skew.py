import cv2
import os
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=5):
    """
    Corrects skew in the input image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        delta (int): Angle delta for determining the range of angles to check (default is 1).
        limit (int): Limit of angles to check, from -limit to limit (default is 5).
    
    Returns:
        tuple: A tuple containing the best angle found for skew correction and the rotated image.
    """
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
          borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def process_skew_correction(input_folder_path, output_folder_path):
    """
    Corrects skew for all images in the input folder and saves the corrected images in the output folder.
    
    Parameters:
        input_folder_path (str): Path to the folder containing input images.
        output_folder_path (str): Path to save the corrected images.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if os.path.isfile(os.path.join(input_folder_path, f))]

    # Process and save images
    for image_file in image_files:
        input_path = os.path.join(input_folder_path, image_file)

        # Load the image
        image = cv2.imread(input_path)

        # Correct skew
        angle, rotated = correct_skew(image)

        # Save the rotated image to the output folder
        output_path = os.path.join(output_folder_path, f"rotated_{image_file}")
        cv2.imwrite(output_path, rotated)

        print(f"Rotated image saved: {output_path}")

# Specify the input folder and output folder
#input_folder_path = 'C:\\Users\\Neha\\Documents\\Major_project\\preprocessing\\grayimage'
#output_folder_path = 'C:\\Users\\Neha\\Documents\\Major_project\\preprocessing\\skewimage'

# Process skew correction for all images in the input folder
#process_skew_correction(input_folder_path, output_folder_path)
