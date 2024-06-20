import cv2
import os
import numpy as np
import concurrent.futures

def image_denoising(image):
    image = np.float32(image)
    denoised_image = cv2.bilateralFilter(image, -1, 25, 25)
    denoised_image = np.uint8(denoised_image)
    return denoised_image

def process_image(input_path, output_folder):
    image = cv2.imread(input_path, 0)
    denoised_image = image_denoising(image)
    output_path = os.path.join(output_folder, f"denoised_{os.path.basename(input_path)}")
    cv2.imwrite(output_path, denoised_image)
    print(f"Denoised image saved: {output_path}")

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda x: process_image(x, output_folder), image_files)