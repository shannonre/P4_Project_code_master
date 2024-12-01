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
from skimage.measure import CircleModel, ransac



site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/10j' # images for sample 10j
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJO3'  # images for sample BJ03
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJ06'  # images for sample BJ06

import cv2
import numpy as np

#plt.imshow(cv2.imread(site_images_path))
#plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough'
def hough_transforms(site_images_path, hough_path, dp=1, min_dist=5, param1=10000, param2=10000000, min_radius = 120, max_radius=1000):
    # param 1 defines how many edges are detected using the Canny edge detector (higher vals = fewer edges)
    # param 2 defines how many votes a circle must receive in order for it to be considered a valid circle (higher vals = a higher no. votes needed)
    hough = False
    if hough:
        if not os.path.exists(hough_path):
            os.makedirs(hough_path)
        detected_circles = {}
        for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
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
                    # for x,y,r in circles[0,:]:
                    #     # draws the circles on the colour image
                    #     cv2.circle(color_image, (x,y), r, (255, 0, 0), 1)
                    #     #cv2.circle(color_image, (x,y), 2, (0, 0, 255), 3)
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        #cv2.circle(color_image, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(color_image, center, radius, (255, 0, 0), 1)
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
                plt.savefig(os.path.join(hough_path,f'hough circles{site_ids}.png'))
                plt.show()
        return detected_circles

detected_circles = hough_transforms(site_images_path, hough_path, min_dist=1, param1=75, param2=10, min_radius=0, max_radius=20)
print("Detected circles:", detected_circles)


from skimage.measure import ransac, CircleModel
import numpy as np
import matplotlib.pyplot as plt
import cv2

ransac_path = 'C:/Users/shann/PycharmProjects/P4 Project/ransac'
# fix issue with ransac detecting one large circle that is not there, instead of the smaller circles that exist in the image.
def ransac_circle_detection(site_images_path, ransac_path, canny_threshold1=1,canny_threshold2=10,residual_threshold=2,min_radius=0,max_radius=100, min_samples=3, max_trials=10000):
    ransac_code = False
    if ransac_code:
        if not os.path.exists(ransac_path):
            os.makedirs(ransac_path)
        detected_ransac_circles = {}
        for i, image in enumerate(os.listdir(site_images_path)[0:3], start=1):
            if image.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
                y_coords, x_coords = np.nonzero(edges)
                edge_points = np.column_stack((x_coords, y_coords))
                print(f"No. edge points detected for image {i}: {len(edge_points)}")


                model_class = CircleModel
                try:
                    ransac_model, inliers = ransac(edge_points,model_class, min_samples=min_samples,residual_threshold=residual_threshold,max_trials=max_trials)
                except ValueError as e:
                    print(f"RANSAC failed for site {i}: {e}")
                    continue

                if ransac_model is not None and ransac_model.params is not None:
                    center_x, center_y, radius = ransac_model.params
                    print(f"Detected circle for site {i}: Center=({center_x:.2f}, {center_y:.2f}), Radius={radius:.2f}")

                    if not (min_radius <= radius <= max_radius):
                        print(f"Detected radius is out of the specified bounds for image {i}.")
                        continue
                    cv2.circle(image, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)
                    #cv2.circle(image, (int(center_x), int(center_y)), 2, (0, 0, 255), 3)

                    detected_ransac_circles[i] = (center_x, center_y, radius)
                else:
                    print(f"No circles detected for image {i}.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(8, 8))
                plt.imshow(image_rgb)
                plt.title(f"Detected Circles using RANSAC for Image {i}")
                plt.axis("off")
                #plt.savefig(os.path.join(ransac_path, f'ransac detection {i}'))
                plt.show()

                # place some saving code here

                # Return the model parameters if needed
        return detected_ransac_circles

ransac_circle_detection(site_images_path, ransac_path, residual_threshold=1, min_samples=3, max_trials=10000, canny_threshold1=10, canny_threshold2=100)

sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results"

def sift_feature_detection(site_images_path, sift_results_path):
    sift = False
    if sift:
        tgt_features = []
        if not os.path.exists(sift_results_path):
            os.makedirs(sift_results_path)

        sift = cv2.SIFT_create(nfeatures=1000,         # max no. features detected
        contrastThreshold=0.03,  # lower vals = detect features with less contrast
        edgeThreshold=5  ,      # lower vals to detect edge like features
        sigma=0.5               # gaussian blur. lower vals = finer features detected
    )

        for i, image_name in enumerate(os.listdir(site_images_path), start=1):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                keypoints, descriptors = sift.detectAndCompute(gray, None)

                print(f"Site {i}: detected {len(keypoints)} keypoints")
                feature_count = 0
                for kp in keypoints:
                    x_coord, y_coord = kp.pt
                    tgt_features.append((int(y_coord), int(x_coord)))
                    feature_count += 1
                    print(f'working on feature {feature_count}')

                print(f"No. target features found in site {i}: {feature_count}")

                keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                keypoint_image_rgb = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 10))
                plt.imshow(keypoint_image_rgb)
                plt.title(f"SIFT Features for Site {i}")
                plt.axis("off")
                plt.savefig(os.path.join(sift_results_path, f"sift_features_image_{i}.png"))
                plt.show()

                descriptors_path = os.path.join(sift_results_path, f"descriptors_image_{i}.npy")
                np.save(descriptors_path, descriptors)
        #print(f' total no. target features = {tgt_features}')
        return tgt_features


tgt_features = sift_feature_detection(site_images_path, sift_results_path)





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
# spectra_results_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_results/BJ03'
# spectra_results_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_results/BJ06'


relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/10j"
# relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/BJ03"
# relative_abundance_path = "C:/Users/shann/PycharmProjects/P4 Project/relative abundances/BJ06"

def compare_peaks(wavelengths, spectrum, known_lines, feature_ids): # add site_ids in here
    compare = True
    if compare:
        peaks, _ = find_peaks(spectrum, prominence=30, threshold=5, height=(2000,250000))
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

        color_map = plt.colormaps.get_cmap('tab20')  #get_cmap('tab10')  # change to matplotlib.colormaps.get_cmap('tab10')
        #substance_colors = {substance: color_map(i) for i, substance in enumerate(known_lines.keys())}
        identified_substances = {}

        for substance, lines in known_lines.items():
            for line in lines:
                if any(abs(detected_wavelengths - line)/line < 0.01):
                    matches = np.where(np.abs(detected_wavelengths - line)/line < 0.01)[0]
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


                        substance_color = color_map(len(identified_substances) / len(known_lines))
                        plt.axvline(x=line, linestyle="--", color=substance_color, linewidth=1, label=f"{substance}")


        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum with Detected Peaks for Feature {feature_ids}") # add site_ids here
        plt.legend()
        #plt.savefig(os.path.join(spectra_results_path, f"spectrum_analysis_site_{site_ids}" ))
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
            print(relative_abundance_data)
            all_data.append(relative_abundance_data)

    final_substance_dataframe = pd.DataFrame({
        "Substance": np.concatenate([data["Substance"] for data in all_data]),
        "Wavelength (nm)": np.concatenate([data["Wavelength (nm)"] for data in all_data]),
        "Intensity": np.concatenate([data["Intensity"] for data in all_data]),
        "Relative Abundance (%)": np.concatenate([data["Relative Abundance (%)"] for data in all_data])
    })
    print(final_substance_dataframe)

        #return identified_substances

known_lines = {
    'H-\u03B1 ': [656.28], # H-alpha, n = 3 to n = 2
    'H-\u03B2 ':[486.13], # h beta, n =4 to n =2
    'H-\u03B3 ' : [434.05],  # H gamma n=5 to n=2  the visible emission lines in the balmer series
    'He': [587.56, 667.82, 447.15],
    'O$_2$': [630.0, 636.4],
    'Na D1 E':[588.9950],
    'Na D2 E': [589.5924], # sodium doublet lines, emission lines from the 3p to 3s transitions in a NA atom
    'K E':[766.5,770.1],  # emission lines in potassium, typically from the 4p → 4s transitions.
    'Ca H A':[396.8],
    'Ca K A': [393.4],  # calcium H and K lines
    'Mg':[285.2,518.4],
    'N$_2$':[337.1,775.3,868.3],
    'H$_2$0 ':[720,820,940,1130], #rotational and vibrational transitions in water vapor.
    'S':[469.4,545.4,605.1],
    'Cl$_2$':[258],
    'Fluorescein E' : [512],  #$C_{20}H_{12}O_5$
    'Fluorescein A ' : [498],
    'Chlorophyll':[685, 740],  # $C_55H_72MgN_4O_5$
    'Chlorophyll a E':[670],
    'Chlorophyll b E': [650],
    'DAPI A':[358],
    'DAPI E':[461],
    'Cy3 A':[550],
    'Cy3 E':[570],
    'Texas Red A':[595],
    'Texas Red E': [615],
    'Rhodamine 6G A':[530],
    'Rhodamine 6G E': [550],
    }


#def relative_abundances():

sites1 = True
if sites1:
    site_counter = 1
    for file, wavelengths, spectrum in handle_multiple_files(spectra_path):
        print(f' processing file {file}')
        compare_peaks(wavelengths, spectrum, known_lines, feature_ids=site_counter)
        site_counter +=1



#from scipy.integrate import simps

# def find_concentrations(wavelengths, spectrum):
#     concentration = False
#     if concentration:
#     # Integrate area under the peak
#     peak_region = (wavelengths >= 490) & (wavelengths <= 510)
#     area = simps(absorbance[peak_region], wavelengths[peak_region])
#
#     # Calibration curve example
#     calibration_concentrations = [0.1, 0.2, 0.3, 0.4, 0.5]  # Known concentrations
#     calibration_areas = [10, 20, 30, 40, 50]  # Corresponding areas under curve
#     slope = np.polyfit(calibration_areas, calibration_concentrations, 1)
#
#     # Determine concentration of unknown sample
#     unknown_concentration = slope[0] * area + slope[1]
#     print(f"Calculated concentration: {unknown_concentration:.2f} M")
#
#     # Plot
#     plt.plot(wavelengths, absorbance)
#     plt.title("Simulated Spectrum")
#     plt.xlabel("Wavelength (nm)")
#     plt.ylabel("Absorbance")
#     plt.show()
