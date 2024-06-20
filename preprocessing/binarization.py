import numpy as np
from PIL import Image
import os

def preprocess_images(folder_path, output_folder, thresh=128):
    """
    Preprocess all images in the folder: perform binarization, skew correction, and bilateral filtering.
    
    Parameters:
        folder_path (str): Path to the folder containing input images.
        output_folder (str): Path to save the preprocessed images.
        thresh (int): Threshold value for binarization (default is 128).
    """
    # Function for binarization
    def binarization_function(input_path, output_path, thresh=128):
        """
        Perform binarization on the input image and save the binarized image to the output path.
        
        Parameters:
            input_path (str): Path to the input image file.
            output_path (str): Path to save the binarized image.
            thresh (int): Threshold value for binarization (default is 128).
        """
        # Load the image and convert it to grayscale
        im_gray = np.array(Image.open(input_path).convert('L'))

        # Perform binarization
        im_bin_keep = (im_gray > thresh) * im_gray
        
        # Save the binarized image to the output folder
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the output folder exists
        Image.fromarray(np.uint8(im_bin_keep)).save(output_path)

        print(f"Binarized image saved: {output_path}")

    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Define output path for the binarized image
        output_path = os.path.join(output_folder, f"binarized_{image_file}")

        # Perform binarization
        binarization_function(image_path, output_path)

# Example usage:
#folder_path = 'grid_remove'  # Replace this with the path to your folder
#output_folder = 'preprocessing\\grayimage'
#preprocess_images(folder_path, output_folder)
