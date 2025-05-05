from PIL import Image
import numpy as np
import os # Import the os module for path operations

#Use this cropping code. Write a for loop to run through all converted png images. This will crop the data

def process_image(input_path, output_path):
    """
    Reads a PNG image, calculates the average color of each pixel column,
    removes columns with an average color of black (0, 0, 0),
    and saves the resulting image.

    Args:
        input_path (str): Path to the input PNG file.
        output_path (str): Path to save the processed PNG file.
    """
    try:
        # 1. Read in the PNG file
        img = Image.open(input_path)

        # Convert image to RGB if it has an alpha channel (RGBA) or is grayscale (L)
        # This simplifies the average calculation. Alpha channel is ignored.
        if img.mode == 'RGBA':
            # Create a white background image
            bg = Image.new("RGB", img.size, (255, 255, 255))
            # Paste the RGBA image onto the white background
            bg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            img_rgb = bg
        elif img.mode == 'LA' or img.mode == 'P': # Handle grayscale with alpha or palette
             img_rgb = img.convert('RGB')
        elif img.mode == 'L': # Handle grayscale
             img_rgb = img.convert('RGB')
        else:
             img_rgb = img # Assume it's already RGB or similar

        # Convert the image to a NumPy array
        img_array = np.array(img_rgb, dtype=np.float32) # Use float for accurate averaging

        # Check if the image is effectively empty after conversion
        if img_array.size == 0 or img_array.shape[1] == 0:
             print(f"Warning: Image '{input_path}' appears empty or has zero width after conversion.")
             return

        # 2. Find the average color of all columns
        # axis=0 calculates the mean along the rows (height dimension) for each column
        # This gives an array where each element is the average [R, G, B] for a column
        average_colors_per_column = np.mean(img_array, axis=0)

        # 3. Remove columns that have an average color of 0 (black)
        # We check if the sum of the average RGB values for a column is greater than a small threshold
        # This avoids potential floating-point inaccuracies where a color might be *very* close to black.
        threshold = 1e-5
        non_black_columns_mask = np.sum(average_colors_per_column, axis=1) > threshold

        # Filter the original image array to keep only non-black columns
        # We apply the mask along the columns axis (axis=1)
        filtered_img_array = img_array[:, non_black_columns_mask, :]

        # Check if any columns remain
        if filtered_img_array.shape[1] == 0:
            print(f"Warning: All columns were removed from '{input_path}'. No output generated.")
            return

        # Convert the filtered array back to an image
        # Ensure the data type is correct for image saving (uint8)
        filtered_img = Image.fromarray(filtered_img_array.astype(np.uint8))

        # Save the processed image
        filtered_img.save(output_path)
        print(f"Processed image saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

#Use this 