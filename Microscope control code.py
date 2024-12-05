#from distutils.command.install import value

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

# folder where images of sites and spectra are saved
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/10j2' # images for sample 10j
#site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJ032'  # images for sample BJ03
site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images/BJ062'  # images for sample BJ06
os.makedirs(site_images_path, exist_ok=True)
site_images_2 = 'C:/Users/shann/PycharmProjects/P4 Project/site_images_axes'
os.makedirs(site_images_2, exist_ok=True)
#actual_site_images_path # where all of the actual site images go
#os.makedirs(site_images_path, exist_ok=True)
#spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/10j2'
#spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/BJ032'
spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra/BJ062'
os.makedirs(spectra_path, exist_ok=True)


# MICROSCOPE IDENTIFIERS
microscope = ofm_client.find_first_microscope()



rows = 1
cols = 4
step_size = 832

#Goes through 3 rows and 3 cols for a total of 9 images with a step size of x or y = 1000 per row. Also performs autofocus at each point.


# POSITION CHECK
def position():
    check_position = True
    #def check_position_code:
    if check_position:
        pos = microscope.position
        pos_array = microscope.get_position_array()
        print(pos_array, 'this is the position array')
        starting_pos = pos.copy() # takes current position of the microscope and stores in a variable (unchanged)
        pos['x'] += 100
        microscope.move(pos) # moves 100 units (pixel units??) along the x-axis
        assert microscope.position == pos, f"X-axis positioning failed"  # checks if microscope has moved as requested
        pos['x'] -= 100
        microscope.move(pos), f"X-axis return failed" # moves microscope back to its starting position

        pos['y'] += 100
        microscope.move(pos)  # Move 100 units along the y-axis
        assert microscope.position == pos, f"Y-axis positioning failed."
        pos['y'] -= 100
        microscope.move(pos)  # Move back to the starting position
        assert microscope.position == starting_pos,f"Y-axis return failed"  # add expected and received sections

        assert microscope.position == starting_pos
        if microscope.position==starting_pos:
            print('positioning is ok')
        else:
            raise ValueError(f'positioning is incorrect, expected {starting_pos} but received {microscope.position}')
            # return
position()


# NEW AUTOFOCUS CODE
def autofocus_and_image(microscope, site_images_path, rows, cols, step_size): #, site_ids):
    os.makedirs(site_images_path, exist_ok=True)
    #print('calibrating microscope stage position with camera...')
    #microscope.calibrate_xy()    # THIS IS NOT WORKING?????????????????????????? figure out why this is ASAP. - works now (29/11/24)
    #print('calibration complete')
    imaging = True
    if imaging:
        for row in range(rows):
            for column in range(cols):
                image_filename = f'Image_row{row}_col{column}.png' #_site id:{site_ids}.png'
                image_filepath = os.path.join(site_images_path, image_filename)

                if os.path.exists(image_filepath):
                    print(f'skipping file {image_filepath} as it already exists')
                    continue
                #else:
                    #plt.imread(image_filepath)
                    #plt.title()
                    #plt.savefig(image_filepath, f'Image_row{row}_col{column}.png')

                x_position = column * step_size
                y_position = row * step_size
                microscope.move_in_pixels(x_position, y_position)
                pos_array = microscope.get_position_array()  # returns the positions
                tgt_position = {'x': x_position, 'y': y_position, 'z':pos_array[2]}

                #microscope.move(tgt_position)
                print('autofocus...')
                microscope.autofocus()

                print('image capture...')
                image = microscope.capture_image()
                image.save(image_filepath)
                plt.imread(image_filepath)

                plt.imshow(image)
                print(f'image saved to {image_filepath}')
        print('scanning completed')


# DISPLYING & SAVING AN IMAGE
def image_looping(site_images_path): # do i need this?? - turn on when you want the scale to be oevrlaid.
    # reads each image in site_file_path, adds a title and axes and saves to a new folder
    image_loop = False
    if image_loop:
        print(f"the number of images in site_images is {len(os.listdir(site_images_path))}")
        #for i, image in enumerate(site_images):
        for site_ids, site_file_name in enumerate(sorted(os.listdir(site_images_path)), start=1):
            site_file_path = os.path.join(site_images_path, site_file_name)
            image = cv2.imread(site_file_path)
            print(f"working on image no.{site_ids} of {len(os.listdir(site_images_path))}")
            #plt.imshow(np.array(image))
            plt.title(f'Image of Site No. {site_ids}')
            #plt.xlabel('label this')
            #plt.ylabel('label this')

            sensor_width = 3680  # in microns
            sensor_height = 2760  # in microns
            sensor_resolution = (3280, 2464)
            image_resolution = (832, 624)
            magnification = 40
            fov_width = sensor_width / magnification
            fov_height = sensor_height / magnification
            scale_x = fov_width / image_resolution[0]
            scale_y = fov_height / image_resolution[1]
            print(f"Horizontal scale: {scale_x:.3f} microns/pixel")
            print(f"Vertical scale: {scale_y:.3f} microns/pixel")

            scale_microns_per_pixel = 0.111  # number of microns per pixel
            scale_bar_length_microns = 10
            scale_bar_thickness = 5
            color = (0,0,0)
            position = (50, 50)
            scale_bar_length_pixels = int(scale_bar_length_microns / scale_microns_per_pixel)
            cv2.rectangle(
                image,
                position,  # Top-left corner of the scale bar
                (position[0] + scale_bar_length_pixels, position[1] + scale_bar_thickness),  # Bottom-right corner
                color,
                -1,  # Thickness (-1 fills the rectangle)
                 )
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label = f"{scale_bar_length_microns} µm"
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_position = (
                position[0],
                position[1] - 10,  # Position above the scale bar
            )

            cv2.putText(image, label, text_position, font, font_scale, color, font_thickness)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


            images_path_2 = os.path.join(site_images_2, f'Image_of_site {site_ids}.png')
            plt.savefig(images_path_2)
            #plt.show()
            #plt.close()
            #print(f"image {site_ids} saved to {images_path_2}")
    #return image_loop

image_loop = image_looping(site_images_path)

# FEATURE IDENTIFICATION

#canny_path = 'C:/Users/shann/PycharmProjects/P4 Project/canny/10j2'
#canny_path = 'C:/Users/shann/PycharmProjects/P4 Project/canny/BJ032'
canny_path = 'C:/Users/shann/PycharmProjects/P4 Project/canny/BJ062'

def canny_edge_detection(site_images_path, canny_path, min_droplet_size=1, max_droplet_size=10000):
    canny1 = True
    all_tgt_features = []
    all_perimeters = []

    if canny1:
        for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
            if image.endswith(('.png', '.jpg')):
                # output_features_file = os.path.join(output_folder, f"features_site_{site_ids}.npy")
                # output_perimeters_file = os.path.join(output_folder, f"perimeters_site_{site_ids}.npy")
                # if os.path.exists(output_features_file) and os.path.exists(output_perimeters_file):
                #     print(f"site {site_ids} already exist. Skipping...")
                #     continue

                image_path = os.path.join(site_images_path, image)
                image_array = cv2.imread(image_path, cv2.COLOR_BGR2GRAY) # cv2.IMREAD_GRAYSCALE
                edges = cv2.Canny(image_array, 1, 200, apertureSize=5)
               # print(f'processing canny edges for image of site id {site_ids}')
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                tgt_features = []
                perimeters = []

                for contour in contours:
                    droplet_area = cv2.contourArea(contour)
                    if min_droplet_size < droplet_area < max_droplet_size:
                        #print(f" droplet area {droplet_area} is ok")
                        M = cv2.moments(contour)  # this gives the contour area
                          # identifies if shape is a closed loop i.e. circle
                        if M['m00'] != 0:
                            x_centroid = int(M['m10'] / M['m00'])
                            y_centroid = int(M['m01'] / M['m00'])  # x and y centroids
                            perimeter = cv2.arcLength(contour, True)
                            # print(f"x centroid is {x_centroid} and the y centroid is {y_centroid}")
                            tgt_features.append((y_centroid, x_centroid))
                            perimeters.append(perimeter)
                            #print(f'features for site {site_ids}: {tgt_features}')
                            #print(f'perimeter for site{site_ids}: {perimeters}')

                        else:
                            print('contour has no area and has been skipped')

                all_tgt_features.append(tgt_features)
                all_perimeters.append(perimeters)

                fig, axs = plt.subplots(1,2, figsize=(10,5))
                axs[0].imshow(image_array, cmap='gray')
                axs[0].set_title(f'Original Image for Site {site_ids}')
                axs[1].imshow(edges, cmap='gray')
                axs[1].set_title(f'Canny Edge Detection for Site {site_ids}')


                #for y,x in tgt_features:
                    #axs[1].scatter(x, y, color='red', s=10)
                #plt.show()
                plt.savefig(os.path.join(canny_path, f'canny_edges_site{site_ids}'))
                #plt.show()
                # plt.close()



    return all_tgt_features, all_perimeters

features, perimeters = canny_edge_detection(site_images_path, canny_path, min_droplet_size=10, max_droplet_size=1e15)

# HOUGH
#hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough/10j2'
#hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough/BJ02'
hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough/BJ02'
def hough_transforms(site_images_path, hough_path, dp=1, min_dist=5, param1=10000, param2=10000000, min_radius = 120, max_radius=1000):
    # param 1 defines how many edges are detected using the Canny edge detector (higher vals = fewer edges)
    # param 2 defines how many votes a circle must receive in order for it to be considered a valid circle (higher vals = a higher no. votes needed)
    hough = True
    if hough:
        tgt_features = []
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
                    for x,y,r in circles[0,:]:
                        # draws the circles on the colour image
                        cv2.circle(color_image, (x,y), r, (255, 0, 0), 1)
                        tgt_features.append((y,x))
                        #cv2.circle(color_image, (x,y), 2, (0, 0, 255), 3)
                #cv2.imshow(f'Detected Circles Using Hough Transforms for Site {site_ids}', color_image)
                # else:
                #     print(f"No circles detected for site {site_ids}.")
                #     detected_circles[site_ids] = []


                #output_file = os.path.join(hough_path, f'{site_ids}.png')
                #cv2.imwrite(output_file, color_image)
                #print(f'hough transform image {site_ids} saved to {hough_path}')

                plt.figure(figsize=(8, 8))
                #plt.imshow(color_image)
                plt.legend()
                plt.title(f"Detected Features via Hough Transforms for Site {site_ids}")
                plt.axis("off")  # Hide axes for better visualization
                plt.savefig(os.path.join(hough_path,f'hough circles{site_ids}.png'))
                #plt.show()
        return detected_circles, tgt_features

detected_circles, tgt_features_hough = hough_transforms(site_images_path, hough_path, min_dist=1, param1=75, param2=10, min_radius=0, max_radius=20)
#print("Detected circles:", detected_circles)
#print("Detected circles:", detected_circles)
#print(f' Hough target features are... {tgt_features_hough}')
print(f' The number of Hough target features are {len(tgt_features_hough)}')



#sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results/10j2"
#sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results/BJ032"
sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results/BJ062"
def sift_feature_detection(site_images_path, sift_results_path):
    sift = True
    if sift:
        tgt_features_sift = []
        sift_descriptors = []
        if not os.path.exists(sift_results_path):
            os.makedirs(sift_results_path)

        sift = cv2.SIFT_create(nfeatures=20000,         # max no. features detected
        contrastThreshold=0.03,  # lower vals = detect features with less contrast
        edgeThreshold=5,        # lower vals to detect edge like features
        sigma=0.5               # gaussian blur. lower vals = finer features detected
    )

        for i, image_name in enumerate(os.listdir(site_images_path), start=1):
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
                plt.savefig(os.path.join(sift_results_path, f"sift_features_site_{i}.png"))
                #plt.show()

                descriptors_path = os.path.join(sift_results_path, f"descriptors_site_{i}.npy")
                np.save(descriptors_path, descriptors)
        #print(f' total no. target features = {tgt_features_sift}')
        return tgt_features_sift, sift_descriptors


tgt_features_sift, sift_descriptors = sift_feature_detection(site_images_path, sift_results_path)
#print(f'SIFT target features are {tgt_features_sift}')
print(f'The number of SIFT target features are {len(tgt_features_sift)}')
print(f'The number of SIFT descriptors are {len(sift_descriptors)}')


# finding common coordinates...
hough_set = set(tgt_features_hough)
sift_set = set(tgt_features_sift)
common_coords = list(hough_set.intersection(sift_set))
common_coords_list = list(common_coords)
print(f'The number of common coordinates is {len(common_coords)}')
#filtered_coords = [(y, x) for y, x in tgt_features_sift if (y, x) in common_coords]
#print(f'the number of filtered coords is {len(filtered_coords)}')

background_spectra_path = "C:/Users/shann/PycharmProjects/P4 Project/background"
def gather_background_spectrum(spectrometer,background_spectra_path):
    background = True
    if background:
        os.makedirs(background_spectra_path, exist_ok=True)
        background_spectrum = spectrometer.spectrum()
        wavelengths = spectrometer.wavelengths()
        intensities= spectrometer.intensities()
        background_save_location = os.path.join(background_spectra_path, "background_spectrum_real2.csv")
        np.savetxt(background_save_location, np.column_stack((wavelengths,intensities)), delimiter=',', header='wavelength,intensity')
        print(f'background spectrum saved to {background_save_location}')



# SPECTRUM GATHERING CODE
# change features depending on canny/hough/ransac
# iterate through a number of features.....

def gather_spectra(spectrometer, spectra_path, common_coords):
    spectra = True
    if spectra:
        os.makedirs(spectra_path, exist_ok=True)

        for feature_ids, site_coords in enumerate(common_coords[0:50], start=1):
            try:
                spectrum = spectrometer.spectrum()
                wavelengths = spectrum[0] # extracting wavelengths
                intensities = spectrum[1] # extracting intensities

                spectra_save_location = os.path.join(spectra_path, f'spectra of features for feature{feature_ids}.csv') #CHANGE THIS DEPENDING ON canny/hough/ransac/sift
                np.savetxt(spectra_save_location, np.column_stack((wavelengths,intensities)), delimiter=',', header='wavelength,intensity')
                print(f' spectrum {feature_ids} saved to {spectra_save_location}')
            except Exception as e:
                print(f' error when saving spectrum {feature_ids}, {e}')

def connect_to_spectrometer():
    devices = sb.list_devices()
    if devices:
        spectrometer = sb.Spectrometer(devices[0])
        print('Spectrometer is on')
        gather_background_spectrum(spectrometer, background_spectra_path)
        return spectrometer
    else:
        print("No spectrometer found")
        return None



def spectra_of_sites(microscope, spectrometer, spectra_path):
    spectra2 = True
    if spectra2:

        #features, perimeters = canny_edge_detection(site_images_path, canny_path, min_droplet_size=10, max_droplet_size=1e15)
        # UN-COMMENT THESE
        #detected_circles, features = hough_transforms(site_images_path, hough_path, min_dist=1, param1=75,param2=10, min_radius=0, max_radius=20)
        #detected_ransac_circles, features = ransac_circle_detection(site_images_path, ransac_path)
        #features = common_coords_list #sift_feature_detection(site_images_path, sift_results_path)

        for feature_ids, site_coords in enumerate(common_coords[0:50], start=1):  # takes all spectra for all coords shared in tgt features and hough transforms

            current_pos = microscope.position.copy()
            tgt_pos = current_pos.copy()
            tgt_pos['x'] += site_coords[0]
            tgt_pos['y'] += site_coords[1]
            microscope.move(tgt_pos)
            #autofocus_and_image(microscope, site_images_path, site_ids)
            #print(f'image displayed of site {site_ids}')
            gather_spectra(spectrometer, spectra_path, common_coords)
            print(f' the number of features is {len(feature_ids)}')
            #print(f'spectra saved of feature {feature_ids}')

#spectra_of_sites(microscope, spectrometer, spectra_path)

#spectrometer.close()

from scipy.signal import find_peaks


def csv_to_png(csv_inputs, png_outputs):
    csv = False
    if csv:
        #if not os.path.exists(png_outputs):
        os.makedirs(png_outputs, exist_ok=True)
        for feature_ids, filename in enumerate(os.listdir(csv_inputs), start=1):
            if filename.endswith('.csv'):
                try:
                    csv_filepath =  os.path.join(csv_inputs, filename)
                    png_filepath = os.path.join(png_outputs, filename.replace('.csv', '.png'))
                    numpy_data = np.loadtxt(csv_filepath, delimiter=',', skiprows=1)
                    wavelengths = numpy_data[:,0]
                    intensities = numpy_data[:,1]
                    peaks, properties = find_peaks(intensities, height=10000)
                    peak_wavelengths = wavelengths[peaks]  # xxx.iloc[peaks]???
                    peak_intensities = intensities[peaks]



                    plt.plot(wavelengths, intensities)
                    plt.scatter(peak_wavelengths, peak_intensities, color='red', marker='x', label='Detected Peaks')
                    #plt.title(f'Background Spectrum')
                    #plt.title(f'Spectrum for Feature {feature_ids}')
                    plt.xlabel(f'Wavelength (nm)')
                    plt.ylabel(f'Intensity')   # potential units (Wm^-2 nm^-1)')
                    for x, y in zip(peak_wavelengths, peak_intensities):
                        plt.text(x, y, f'{x:.1f} nm', fontsize=8, ha='right', va='bottom')

                    plt.savefig(png_filepath, format='png')
                    plt.show()
                    #print('displayed background png')
                    #print(f"spectra saved of feature {feature_ids}")
                    #plt.close()
                except Exception as e:
                    print(f'error is {e}')


if __name__ == "__main__":
    position()
    autofocus_and_image(microscope, site_images_path, rows, cols, step_size) #, site_ids)
    image_looping(site_images_path)
    #masks = color_masking(site_images_path)
    #features = size_masking(masks[0])
    spectrometer = connect_to_spectrometer()
    if spectrometer:
        print(f'connected to spectrometer')
        gather_spectra(spectrometer, spectra_path, common_coords)

#spectra_of_sites(microscope, spectrometer, spectra_path)
#png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/10j2'
#png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/BJ032'
#png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png/BJ062'
background_png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/background'
#csv_to_png(spectra_path, png_output_folder)
#csv_to_png(background_spectra_path, background_png_output_folder)



# check new common coords on 5th dec






