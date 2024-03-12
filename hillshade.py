import numpy as np
import os
from PIL import Image, ImageChops

def read_hgt_file(file_path):
    with open(file_path, 'rb') as file:
        # Assuming SRTM1 HGT format (16-bit signed integers, big-endian)
        data = np.frombuffer(file.read(), dtype='>i2').reshape((1201, 1201)) * 3.28
    return data

def create_shaded_relief(elevation_data, azimuth=100, elevation=45):
    # Calculate gradient using Sobel filter
    gradient_x = np.gradient(elevation_data, axis=1)
    gradient_y = np.gradient(elevation_data, axis=0)

    # Calculate hillshade intensity
    hillshade = (np.sin(np.radians(elevation)) * np.sin(np.radians(gradient_y))
                 + np.cos(np.radians(elevation)) * np.cos(np.radians(gradient_y))
                 * np.cos(np.radians(azimuth - 90 - gradient_x)))

    # Normalize to [0, 255] for image representation
    normalized_hillshade = ((hillshade - hillshade.min()) / (hillshade.max() - hillshade.min()) * 255).astype(np.uint8)
    return normalized_hillshade

def create_shaded_relief_with_alpha(elevation_data, azimuth=300, elevation=40):
    # Calculate gradient using Sobel filter
    gradient_x = np.gradient(elevation_data, axis=1)
    gradient_y = np.gradient(elevation_data, axis=0)

    # Calculate hillshade intensity
    hillshade = (np.sin(np.radians(elevation)) * np.sin(np.radians(gradient_y))
                 + np.cos(np.radians(elevation)) * np.cos(np.radians(gradient_y))
                 * np.cos(np.radians(azimuth - 90 - gradient_x)))

    # Normalize to [0, 1] for alpha channel
    normalized_hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())

    # Create RGBA array with alpha channel representing brightness
    shaded_relief_rgba = np.zeros((elevation_data.shape[0], elevation_data.shape[1], 4), dtype=np.uint8)
    shaded_relief_rgba[:, :, 0] = normalized_hillshade * 160  # Red channel
    shaded_relief_rgba[:, :, 1] = normalized_hillshade * 160  # Green channel
    shaded_relief_rgba[:, :, 2] = normalized_hillshade * 160  # Blue channel
    shaded_relief_rgba[:, :, 3] = (255-normalized_hillshade * 128).astype(np.uint8)  # Alpha channel

    return shaded_relief_rgba

def create_colorful_elevation_map(elevation_data):
    # Define elevation ranges and corresponding colors
    elevation_ranges = [0, 1, 500, 1000, 1500, 2000, 2500, 4000, 6000, 9000, 12000, 20000, 40000, np.inf]
    colors = [
        (17, 54, 115),
        (85, 221, 68),
        (51, 187, 51),
        (0, 136, 34),
        (0, 85, 17),
        (0, 68, 17),
        (221, 153, 68),
        (187, 136, 51),
        (153, 102, 17),
        (136, 85, 17),
        (102, 68, 0),
        (68, 34, 0),
        (238, 238, 221),
        (238, 238, 221)
    ]
    # Create an empty RGB image
    rgb_image = np.zeros((1201, 1201, 3), dtype=np.uint8)

    # Assign colors based on elevation ranges
    for i in range(len(elevation_ranges) - 1):
        mask = np.logical_and(elevation_data >= elevation_ranges[i], elevation_data < elevation_ranges[i + 1])
        rgb_image[mask] = colors[i]
    return rgb_image

def overlay_images(background_path, overlay_path, output_path, alpha=0.5):
    # Open the background and overlay images
    background = Image.open(background_path)
    overlay = Image.open(overlay_path)

    # Resize overlay image to match the background size
    overlay = overlay.resize(background.size, Image.LANCZOS)

    # Prepare overlay image with alpha channel
    overlay = overlay.convert("RGBA")
    overlay_with_alpha = Image.new("RGBA", overlay.size)
    overlay_with_alpha = Image.blend(overlay_with_alpha, overlay, alpha)
    over_multiply = ImageChops.multiply(overlay, overlay)
    over_multiply = over_multiply.convert("RGB")
    over_multiply.save(output_path, "PNG")
    return 
    # Composite the images
    merged_image = Image.alpha_composite(background.convert("RGBA"), overlay_with_alpha)
    merged_image = merged_image.transpose(Image.FLIP_TOP_BOTTOM)
    merged_image = merged_image.convert("RGB")
    merged_image.save(output_path, "PNG")

def save_image(array, output_path, type):
    Image.fromarray(array, type).save(output_path)

def process_hgt_files(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.hgt'):
                hgt_file_path = os.path.join(foldername, filename)
                elevation_data = read_hgt_file(hgt_file_path)
                #shaded_relief = create_shaded_relief(elevation_data)
                shaded_relief = create_shaded_relief_with_alpha(elevation_data)

                # Extract subfolder information
                rel_path = os.path.relpath(hgt_file_path, root_folder)
                subfolder = os.path.dirname(rel_path)

                # Output file paths in the "topo" folder
                output_image_path = os.path.join('topo', subfolder, f'{os.path.splitext(filename)[0]}_hills.png')
                save_image(shaded_relief, output_image_path, 'RGBA')

                colorful_elevation_map = create_colorful_elevation_map(elevation_data)

                output_image_path = os.path.join('topo', subfolder, f'{os.path.splitext(filename)[0]}_colors.png')
                save_image(colorful_elevation_map, output_image_path, 'RGB')

                background_image_path = os.path.join('topo', subfolder, f'{os.path.splitext(filename)[0]}_hills.png')
                overlay_image_path = os.path.join('topo', subfolder, f'{os.path.splitext(filename)[0]}_colors.png')
                output_image_path = os.path.join('topo', subfolder, f'{os.path.splitext(filename)[0]}.png')

                overlay_images(background_image_path, overlay_image_path, output_image_path)

# Assuming the functions read_hgt_file, create_shaded_relief, save_image,
# create_colorful_elevation_map, and overlay_images are defined elsewhere.

# Running the script
process_hgt_files('hgt')

exit(0)
hgt_file_path = 'hgt\\S24W046.hgt'
elevation_data = read_hgt_file(hgt_file_path)
shaded_relief = create_shaded_relief(elevation_data)

output_image_path = 'S24W046_hills.png'
save_image(shaded_relief, output_image_path, 'L')

colorful_elevation_map = create_colorful_elevation_map(elevation_data)

output_image_path = 'S24W046_colors.png'
save_image(colorful_elevation_map, output_image_path, 'RGB')

background_image_path = 'S24W046_hills.png'
overlay_image_path = 'S24W046_colors.png'
output_image_path = 'S24W046.png'

overlay_images(background_image_path, overlay_image_path, output_image_path)
