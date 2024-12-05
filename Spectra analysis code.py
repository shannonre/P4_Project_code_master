import os
import cv2
import imageio.v3 as iio
import openflexure_microscope_client as ofm_client
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.pyplot import contour
import seabreeze
import seabreeze.spectrometers as sb
from semantic_version import compare
from skimage.measure import CircleModel, ransac



site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/10j' # images for sample 10j
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJ03'  # images for sample BJ03
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJ06'  # images for sample BJ06

import cv2
import numpy as np

#plt.imshow(cv2.imread(site_images_path))
#plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough/10j'
def hough_transforms(site_images_path, hough_path, dp=1, min_dist=5, param1=10000, param2=10000000, min_radius = 120, max_radius=1000):
    # param 1 defines how many edges are detected using the Canny edge detector (higher vals = fewer edges)
    # param 2 defines how many votes a circle must receive in order for it to be considered a valid circle (higher vals = a higher no. votes needed)
    hough = True
    if hough:
        tgt_features_hough = []
        if not os.path.exists(hough_path):
            os.makedirs(hough_path)
        detected_circles = {}
        for site_ids, image in enumerate(os.listdir(site_images_path)[0:1], start=1):
            if image.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image)
                color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                #gaussian_blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

                rows = grayscale_image.shape[0]
                circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=rows/108, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    detected_circles[site_ids] = circles[0, :]
                    for x, y, r in circles[0, :]:
                        # draws the circles on the colour image
                        cv2.circle(color_image, (x, y), r, (255, 0, 0), 1)
                        tgt_features_hough.append((y, x))
                #cv2.imshow(f'Detected Circles Using Hough Transforms for Site {site_ids}', color_image)
                # else:
                #     print(f"No circles detected for site {site_ids}.")
                #     detected_circles[site_ids] = []


                #output_file = os.path.join(hough_path, f'{site_ids}.png')
                #cv2.imwrite(output_file, color_image)
                print(f'hough transform image {site_ids} saved to {hough_path}')

                plt.figure(figsize=(8, 8))
                plt.imshow(color_image)
                plt.legend()
                plt.title(f"Detected Features via Hough Transforms for Site {site_ids}")
                plt.axis("off")  # Hide axes for better visualization
                #plt.savefig(os.path.join(hough_path,f'hough circles{site_ids}.png'))
                plt.show()
        return detected_circles, tgt_features_hough

detected_circles, tgt_features_hough = hough_transforms(site_images_path, hough_path, min_dist=1, param1=75, param2=10, min_radius=0, max_radius=20)
print("Detected circles:", detected_circles)
#print(f' Hough target features are... {tgt_features_hough}')
print(f' The number of Hough target features are {len(tgt_features_hough)}')


from skimage.measure import ransac, CircleModel
import numpy as np
import matplotlib.pyplot as plt
import cv2

sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results/10j"

def sift_feature_detection(site_images_path, sift_results_path):
    sift = True
    if sift:
        tgt_features_sift = []
        sift_descriptors = []
        if not os.path.exists(sift_results_path):
            os.makedirs(sift_results_path)

        sift = cv2.SIFT_create(nfeatures=20000,         # max no. features detected
        contrastThreshold=0.03,  # lower vals = detect features with less contrast
        edgeThreshold=5  ,      # lower vals to detect edge like features
        sigma=0.5)              # gaussian blur. lower vals = finer features detected

        for i, image_name in enumerate(os.listdir(site_images_path)[0:1], start=1):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                keypoints, descriptors = sift.detectAndCompute(gray, None)

                #print(f"Site {i}: detected {len(keypoints)} keypoints")
                feature_count = 0
                for kp in keypoints:
                    x_coord, y_coord = kp.pt
                    tgt_features_sift.append((int(y_coord), int(x_coord)))
                    sift_descriptors.append(descriptors)
                    feature_count += 1
                    #print(f'working on feature {feature_count}')

                #print(f"No. target features found in site {i}: {feature_count}")

                keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                keypoint_image_rgb = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 10))
                plt.imshow(keypoint_image_rgb)
                plt.title(f"SIFT Features for Site {i}")
                plt.axis("off")
                #plt.savefig(os.path.join(sift_results_path, f"sift_features_image_{i}.png"))
                plt.show()

                descriptors_path = os.path.join(sift_results_path, f"descriptors_image_{i}.npy")
                #np.save(descriptors_path, descriptors)
        #print(f' total no. target features = {tgt_features}')
        return tgt_features_sift , sift_descriptors


tgt_features_sift, sift_descriptors = sift_feature_detection(site_images_path, sift_results_path)
#print(f'SIFT target features are {tgt_features_sift}')
print(f'The number of SIFT target features are {len(tgt_features_sift)}')
print(f'The number of SIFT descriptors are {len(sift_descriptors)}')


# finding common coordinates...
hough_set = set(tgt_features_hough)
sift_set = set(tgt_features_sift)
common_coords = hough_set.intersection(sift_set)
print(f'The number of common coordinates is {len(common_coords)}')
filtered_sift_coords = [(y, x) for y, x in tgt_features_sift if (x, y) in common_coords]  # to use when gathering spectra





# filtered_sift_coords = [(y, x) for y, x in tgt_features_sift if (x, y) in common_coords]
# for i, image_name in enumerate(os.listdir(site_images_path)[0:1], start=1):
#     if image_name.endswith(('.png', '.jpg')):
#         image_path = os.path.join(site_images_path, image_name)
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         sift_keypoints = [cv2.KeyPoint(x=x, y=y, size=1.0, angle=0, response=0, octave=0, class_id=0) for y, x in
#                           filtered_sift_coords]
#
#         img = cv2.drawKeypoints(image, sift_keypoints, image)
#         plt.figure(figsize=(10, 10))
#         plt.imshow(img)
#         #for y, x in filtered_sift_coords:
#          #   plt.plot(x, y, 'ro', markersize=5)  #red circle with size 5
#         plt.title(f"Image: {image_name}")
#         plt.show()



# # Assuming:
# # - filtered_sift_coords is a list of tuples (y, x)
# # - sift_descriptors is a list of descriptors, corresponding to the coordinates in filtered_sift_coords
# filtered_sift_coords = [(y, x) for y, x in tgt_features_sift if (x, y) in common_coords]
# for i, image_name in enumerate(os.listdir(site_images_path)[2:3], start=1):
#     if image_name.endswith(('.png', '.jpg')):
#         image_path = os.path.join(site_images_path, image_name)
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         sift_keypoints = [cv2.KeyPoint(x=x, y=y, size=1.0, angle=0, response=float(np.mean(descriptor)), octave=0, class_id=0)
#                           for (y, x), descriptor in zip(filtered_sift_coords, sift_descriptors)]
#
#         img = cv2.drawKeypoints(image, sift_keypoints, image)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         plt.figure(figsize=(10, 10))
#         plt.imshow(img)
#         plt.title(f"Image: {image_name}")
#         plt.show()



# SPECTRA ANALYSIS CODE.

spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/10j'
#spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/BJ03'
#spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/BJ06'


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

spectra_results_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_results/10j'
#spectra_results_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_results/BJ03'
#spectra_results_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_results/BJ06'


relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/10j"
# #relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/BJ03"
#relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/BJ06"
from matplotlib.colors import Normalize

def csv_to_png(csv_inputs, png_outputs):
    csv = False
    if csv:
        #if not os.path.exists(png_outputs):
        os.makedirs(png_outputs, exist_ok=True)
        for feature_ids, filename in enumerate(os.listdir(csv_inputs)[0:2], start=1):
            if filename.endswith('.csv'):
                try:
                    csv_filepath =  os.path.join(csv_inputs, filename)
                    png_filepath = os.path.join(png_outputs, filename.replace('.csv', '.png'))
                    numpy_data = np.loadtxt(csv_filepath, delimiter=',', skiprows=1)
                    wavelengths = numpy_data[:,0]
                    intensities = numpy_data[:,1]
                    #peaks, _ = find_peaks(spectrum, prominence=50, threshold=1, height=(2000, 250000))
                    peaks, properties = find_peaks(intensities, height=10000)
                    peak_wavelengths = wavelengths[peaks]  # xxx.iloc[peaks]???
                    peak_intensities = intensities[peaks]


                    plt.plot(wavelengths, intensities)
                    plt.scatter(peak_wavelengths, peak_intensities, color='red', marker='x', label='Detected Peaks')
                    plt.title(f'Spectrum of Feature {feature_ids} with Detected Peaks')
                    plt.xlabel(f'Wavelength (nm)')
                    plt.ylabel(f'Intensity')   # potential units (Wm^-2 nm^-1)')
                    for x, y in zip(peak_wavelengths, peak_intensities):
                        plt.text(x, y + 0.02 * max(peak_intensities), f"{x:.1f} nm", fontsize=8, ha="center",
                                 va="bottom")

                    #plt.savefig(png_filepath, format='png')
                    plt.legend()
                    #plt.show()

                    print(f"spectra displayed of feature {feature_ids}")
                    #plt.close()
                except Exception as e:
                    print(f'error is {e}')


png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/10j'
# png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/BJ03'
# png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/BJ06'
csv_to_png(spectra_path,png_output_folder)

# add relative abundance tables for each substance


def compare_peaks(wavelengths, spectrum, known_lines, feature_ids): # add site_ids in here
    compare = False
    if compare:
        peaks, _ = find_peaks(spectrum, prominence=50, threshold=1, height=(2000,250000))
        detected_wavelengths = wavelengths[peaks]
        peak_intensities = spectrum[peaks]
        peak_wavelengths = detected_wavelengths

        total_intensity = peak_intensities.sum()
        relative_abundances = (peak_intensities / total_intensity) * 100

        # relative_save_location = os.path.join(relative_abundance_path, )
        # relative_abundance_data.to_csv(relative_save_location, index=False)

        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, spectrum, label="Observed Spectrum", color='blue', linewidth=1.5)
        plt.scatter(detected_wavelengths, peak_intensities, color='red', marker='x', label='Detected Peaks')

        # normalise color map to ensure it falls over entire range of available colors
        norm = Normalize(vmin=0, vmax=len(known_lines))
        color_map = plt.colormaps.get_cmap('Set1')  #get_cmap('tab10')  # change to matplotlib.colormaps.get_cmap('tab10')

        #substance_colors = {substance: color_map(i) for i, substance in enumerate(known_lines.keys())}
        identified_substances = {}
        labels = set()

        for substance, lines in known_lines.items():
            for line in lines:
                if any(abs(detected_wavelengths - line)/line < 0.005):
                    matches = np.where(np.abs(detected_wavelengths - line)/line < 0.005)[0]
                    for match in matches:
                        matched_wavelength = detected_wavelengths[match]
                        intensity = spectrum[np.where(wavelengths == matched_wavelength)[0][0]]

                        if substance not in identified_substances:
                            identified_substances[substance] = {
                                "wavelengths": [matched_wavelength],
                                "intensities": [intensity],
                                "relative_abundances": [relative_abundances[match]]}

                        else:
                            identified_substances[substance]["wavelengths"].append(matched_wavelength)
                            identified_substances[substance]["intensities"].append(intensity)
                            identified_substances[substance]["relative_abundances"].append(relative_abundances[match])

                        if substance not in labels:
                            substance_color = color_map(len(identified_substances)) # / len(known_lines))
                            plt.axvline(x=line, linestyle="--", color=substance_color, linewidth=1, label=f"{substance}")
                            labels.add(substance)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum with Detected Peaks for Feature {feature_ids}") # add site_ids here
        plt.legend()
        #plt.savefig(os.path.join(spectra_results_path, f"spectrum_analysis_site_{feature_ids}" ))
        #plt.grid(alpha=0.5)
        plt.show()

        all_data = []

        for substance, data in identified_substances.items():
            #print(f" substance is {substance} intensity is {intensity}")
            relative_abundance_data = pd.DataFrame({
                "Substance": [substance] * len(data["wavelengths"]),
                "Wavelength (nm)": data['wavelengths'],#peak_wavelengths,
                "Intensity": data['intensities'], #peak_intensities,
                "Relative Abundance (%)": data['relative_abundances']}) #relative_abundances})
            #print(relative_abundance_data)
            all_data.append(relative_abundance_data)

        final_substance_dataframe = pd.DataFrame({
            "Substance": np.concatenate([data["Substance"] for data in all_data]),
            "Wavelength (nm)": np.concatenate([data["Wavelength (nm)"] for data in all_data]),
            "Intensity": np.concatenate([data["Intensity"] for data in all_data]),
            "Relative Abundance (%)": np.concatenate([data["Relative Abundance (%)"] for data in all_data])
        })
        print(final_substance_dataframe)

        return #identified_substances

known_lines = {
    'H-\u03B1 ': [656.28], # H-alpha, n = 3 to n = 2
    'H-\u03B2 ':[486.13], # h beta, n =4 to n =2
    'H-\u03B3 ' : [434.05],  # H gamma n=5 to n=2  the visible emission lines in the balmer series
    'He': [587.56, 667.82, 447.15],  # emission lines
    'O$_2$': [630.0, 636.4],  # emission lines
    'Na D1':[588.9950],
    'Na D2': [589.5924], # sodium doublet lines, emission lines from the 3p to 3s transitions in a NA atom
    'K':[766.5,770.1],  # emission lines in potassium, typically from the 4p → 4s transitions.
    'Ca H A':[396.8],
    'Ca K A': [393.4],  # calcium H and K absorption lines
    'Mg':[285.2,518.4], # emission lines
    'N$_2$':[337.1,775.3,868.3], # emission lines
    'H$_2$0 ':[720,820,940,1130], #rotational and vibrational transitions in water vapor.
    'S':[469.4,545.4,605.1], # emission lines
    'Cl$_2$':[258], # emission
    'Fluorescein E' : [512],
    'Fluorescein A ' : [498],
    'Chlorophyll':[685, 740],
    'Chlorophyll a E':[670],
    'Chlorophyll b E': [650],
    'DAPI A':[358],
    'DAPI E':[461],
    'Cy3 A':[554],
    'Cy3 E':[568],
    'Texas Red A':[595],
    'Texas Red E': [615],
    'Rhodamine 6G A':[530],
    'Rhodamine 6G E': [552],
    }




sites1 = False
if sites1:
    site_counter = 1
    for file, wavelengths, spectrum in handle_multiple_files(spectra_path):
        print(f' processing file {file}')
        compare_peaks(wavelengths, spectrum, known_lines, feature_ids=site_counter)
        site_counter +=1

from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches

# edit this later!!
def image_looping(site_images_path, output_path):
    loop = False
    if loop:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(f"Processing images in {site_images_path}...")
        image_files = sorted(os.listdir(site_images_path))
        print(f"Number of images found: {len(image_files)}")

        for site_ids, site_file_name in enumerate(image_files, start=1):
            site_file_path = os.path.join(site_images_path, site_file_name)
            image = cv2.imread(site_file_path)

            if image is None:
                print(f"Skipping {site_file_name}, unable to read the file.")
                continue

            print(f"Working on image {site_ids} of {len(image_files)}: {site_file_name}")

            sensor_width = 3680  # in microns
            sensor_height = 2760  # in microns
            sensor_resolution = (3280, 2464)
            image_resolution = (832, 624)
            magnification = 40

            fov_width = sensor_width / magnification
            fov_height = sensor_height / magnification
            scale_microns_per_pixel = fov_width / image_resolution[0]

            scale_bar_length_microns = 10
            scale_bar_length_pixels = int(scale_bar_length_microns / scale_microns_per_pixel)
            position = (50, image.shape[0] - 50)
            scale_bar_thickness = 5
            scale_bar_color = "black"

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

            scale_bar = patches.Rectangle(
                position, scale_bar_length_pixels, scale_bar_thickness, color=scale_bar_color)
            ax.add_patch(scale_bar)

            mu = "\u03bc"  # Unicode for 'mu'
            label = f"{scale_bar_length_microns} {mu}m"
            label_position = (position[0], position[1] - 30)  # Above the scale bar
            ax.text(
                label_position[0],
                label_position[1],
                label,
                color=scale_bar_color,
                fontsize=12,
                ha="left",
                va="top",)
            # plt.figure(figsize=(8, 8))
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Image of Site {site_ids}")
            plt.axis('off')

            output_file = os.path.join(output_path, f'Image_of_site_{site_ids}_with_scale.png')
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            #plt.close()
            #plt.show()

            #print(f"Saved processed image to {output_file}")

output_path ='C:/Users/shann/PycharmProjects/P4 Project/scale/10j'
#output_path ='C:/Users/shann/PycharmProjects/P4 Project/scale/BJ03'
#output_path ='C:/Users/shann/PycharmProjects/P4 Project/scale/BJ06'
image_looping(site_images_path, output_path)


background_spectra_path = "C:/Users/shann/PycharmProjects/P4 Project/background/background_spectrum4.csv"
noise_subtracted = "C:/Users/shann/PycharmProjects/P4 Project/noise_subtracted"
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fix this later
def subtract_background_from_folder(site_folder, background_csv, output_folder=noise_subtracted, plot=True):
    background = True
    if background:
        background_spectrum = pd.read_csv(background_csv, header=None, names=["Wavelength", "Intensity"],skiprows=1)
        background_spectrum["Wavelength"] = pd.to_numeric(background_spectrum["Wavelength"], errors="coerce")
        background_spectrum["Intensity"] = pd.to_numeric(background_spectrum["Intensity"], errors="coerce")

        background_spectrum = background_spectrum.dropna() # removes any rows with invalid data
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        noise_subtracted_spectra = {}
        for site_file_name in sorted(os.listdir(site_folder)[1:2]):
            site_file_path = os.path.join(site_folder, site_file_name)
            if not site_file_name.endswith(".csv"):
                continue

            site_spectrum = pd.read_csv(site_file_path, header=None, names=["Wavelength", "Intensity"],skiprows=1)


            interpolated_background = np.interp(
                site_spectrum["Wavelength"],
                background_spectrum["Wavelength"],
                background_spectrum["Intensity"])


            subtracted_intensities = site_spectrum["Intensity"] - interpolated_background


            noise_subtracted_spectrum = pd.DataFrame({
                "Wavelength": site_spectrum["Wavelength"],
                "Subtracted Intensity": subtracted_intensities})


            if output_folder:
                output_csv_path = os.path.join(output_folder, f"noise_subtracted_{site_file_name}")
                noise_subtracted_spectrum.to_csv(output_csv_path, index=False)


            noise_subtracted_spectra[site_file_name] = noise_subtracted_spectrum


            if plot:
                plt.figure(figsize=(10, 6))
                #plt.plot(site_spectrum["Wavelength"], site_spectrum["Intensity"], label="Site Spectrum", color="blue")
                #plt.plot(background_spectrum["Wavelength"], background_spectrum["Intensity"], label="Background Spectrum",color="orange")
                plt.plot(noise_subtracted_spectrum["Wavelength"], noise_subtracted_spectrum["Subtracted Intensity"],
                         label="Noise-Subtracted Spectrum", color="green")
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Intensity")
                plt.title(f"Noise-Subtracted Spectrum for {site_file_name}")
                plt.legend()
                plt.grid(alpha=0.5)
                plt.show()

        return noise_subtracted_spectra

site_folder = spectra_path

noise_spectra = subtract_background_from_folder(spectra_path,background_spectra_path,output_path, plot=True)

