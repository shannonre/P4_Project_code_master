import os
import cv2
import imageio.v3 as iio
import openflexure_microscope_client as ofm_client
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import contour
import seabreeze
import seabreeze.spectrometers as sb
from skimage.measure import CircleModel, ransac



site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/Image_0.png'

import cv2
import numpy as np

#plt.imshow(cv2.imread(site_images_path))
#plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def hough_circle_detection(site_images_path,min_dist=10,param1=100,param2=30,min_radius=10,max_radius=1000):
    hough1 = False
    if hough1:
        # Load the image
        image = cv2.imread(site_images_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {site_images_path}")

        # Display the original image using matplotlib (optional for debugging)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.show()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        # Draw detected circles on the image
        if circles is not None:
            circles = np.uint16(np.around(circles))  # Round and convert to integers
            for x, y, r in circles[0, :]:
                # Draw the circle outline
                cv2.circle(image, (x, y), r, (0, 255, 0), 1)
                # Draw the circle's center
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
        else:
            print("No circles detected.")

        # Display the image with detected circles
        #cv2.imshow("Detected Circles", image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.legend()
        plt.title("Detected Circles")
        plt.axis("off")  # Hide axes for better visualization
        plt.show()
        #cv2.waitKey(0)  # Wait for a key press to close the window
        #cv2.destroyAllWindows()



# Example usage
image_path = "path_to_input_image.jpg"
#output_path = "path_to_output_image_with_circles.jpg"
detected_circles = hough_circle_detection(site_images_path,min_dist=10, param1=10, param2=10, min_radius=10, max_radius=1000)
print("Detected circles:", detected_circles)


def sharpen_image(image):
    # Create a kernel for sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the kernel to the image
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
def denoise_image(image):
    # Apply bilateral filtering to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised
def adjust_contrast_and_brightness(image, alpha=1.5, beta=20):
    # alpha is the contrast factor, beta is the brightness offset
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted
def preprocess_image(site_images_path):
    # Load the image
    image = cv2.imread(site_images_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {site_images_path}")

    # Step 1: Denoise
    denoised = denoise_image(image)

    # Step 2: Sharpen
    sharpened = sharpen_image(denoised)

    # Step 3: Adjust Contrast and Brightness
    processed = adjust_contrast_and_brightness(sharpened)

    return processed



from skimage.measure import ransac, CircleModel
import numpy as np
import matplotlib.pyplot as plt
import cv2

def ransac_circle_detection(
    site_images_path,
    canny_threshold1=100,
    canny_threshold2=200,
    residual_threshold=2,
    min_radius=1,
    max_radius=10000):
    ran = False
    if ran:
        # Load the image
        image = cv2.imread(site_images_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at {site_images_path}")

        processed_image = preprocess_image(site_images_path)
        # Convert to grayscale
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

        # Get coordinates of edge points
        y_coords, x_coords = np.nonzero(edges)
        edge_points = np.column_stack((x_coords, y_coords))
        print(f"Number of edge points detected: {len(edge_points)}")

        # Use the CircleModel class from skimage
        model_class = CircleModel

        # Run RANSAC
        try:
            ransac_model, inliers = ransac(
                edge_points,
                model_class,  # Pass the class, not an instance
                min_samples=3,
                residual_threshold=residual_threshold,
                max_trials=1000
            )
        except ValueError as e:
            print(f"RANSAC failed: {e}")
            return

        # Get circle parameters from the model
        if ransac_model is not None and ransac_model.params is not None:
            center_x, center_y, radius = ransac_model.params
            print(f"Detected circle: Center=({center_x:.2f}, {center_y:.2f}), Radius={radius:.2f}")

            if not (min_radius <= radius <= max_radius):
                print("Detected radius is out of the specified bounds.")
                return
        else:
            print("No circles detected.")
            return

        # Draw the detected circle
        cv2.circle(image, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)
        cv2.circle(image, (int(center_x), int(center_y)), 2, (0, 0, 255), 3)

        # Convert the image from BGR to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with detected circle
        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.title("Detected Circle (RANSAC)")
        plt.axis("off")  # Hide axes for better visualization
        plt.show()

        # Return the model parameters if needed
        return center_x, center_y, radius


# Example usage
#site_images_path = "path_to_your_image.png"  # Replace with your image path
ransac_circle_detection(site_images_path)


# SPECTRA ANALYSIS CODE.

spectrum_path = "C:/Users/shann/PycharmProjects/P4 Project/spectra"


import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap


def load_spectra(spectrum_path):
    try:
        df = pd.read_csv(spectrum_path, skiprows=1, names=["Wavelength", "Intensity"])
        df = df.astype({"Wavelength": float, "Intensity": float})
        wavelengths = df['Wavelength'].values
        spectrum = df['Intensity'].values
        return wavelengths, spectrum
    except Exception as e:
        raise RuntimeError(f"error loading spectrum data: {e}")

#wavelengths, spectrum = load_spectra(spectrum_path)

def handle_multiple_files(folder):
    spectra = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            wavelengths, spectrum = load_spectra(file_path)
            spectra.append((file, wavelengths, spectrum))
    return spectra

def compare_peaks(wavelengths, spectrum, known_lines, site_ids): # add site_ids in here
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, spectrum, label="Observed Spectrum", color='blue', linewidth=1.5)

    color_map = get_cmap('tab10')  # change to matplotlib.colormaps.get_cmap('tab10')
    substance_colors = {substance: color_map(i) for i, substance in enumerate(known_lines.keys())}

    peaks, _ = find_peaks(spectrum, prominence=0.1)
    detected_wavelengths = wavelengths[peaks]
    for substance, lines in known_lines.items():
        for line in lines:
            if any(abs(detected_wavelengths - line) < 1):
                plt.axvline(x=line, linestyle="--", color=substance_colors[substance], linewidth=1, label=f"{substance}")
                break

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum with Detected Peaks for Site {site_ids}") # add site_ids here
    plt.legend()
    #plt.grid(alpha=0.5)
    plt.show()

known_lines = {
    'H$_2$': [656.28, 486.13, 434.05],  # H-alpha, H-beta, H-gamma (in nm)
    'He': [587.56, 667.82, 447.15],
    'O$_2$': [630.0, 636.4],
    'Na':[589.0,589.6],
    'K':[766.5,770.1],
    'Ca':[422.7,393.4],
    'Mg':[285.2,518.4],
    'N$_2$':[337.1,775.3,868.3],
    'H$_2$0':[720,820,940,1130],
    'S':[469.4,545.4,605.1],
    'Cl$_2$':[258],
    }

site_counter = 1
for file, wavelengths, spectrum in handle_multiple_files(spectrum_path):
    print(f' processing file {file}')
    compare_peaks(wavelengths, spectrum, known_lines, site_ids=site_counter)
    site_counter +=1
