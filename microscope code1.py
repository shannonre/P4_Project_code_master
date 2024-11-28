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
site_images_path = 'C:/Users/shann/PycharmProjects/P4 Project/site_images' # this is where all my trial images go
os.makedirs(site_images_path, exist_ok=True)
site_images_2 = 'C:/Users/shann/PycharmProjects/P4 Project/site_images_axes'
os.makedirs(site_images_2, exist_ok=True)
#actual_site_images_path # where all of the actual site images go
#os.makedirs(site_images_path, exist_ok=True)
spectra_path = 'C:/Users/shann/PycharmProjects/P4 Project/spectra'
os.makedirs(spectra_path, exist_ok=True)


# MICROSCOPE IDENTIFIERS
microscope = ofm_client.find_first_microscope()

rows = 20
cols = 20
step_size = 1000


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
        assert microscope.position == pos  # checks if microscope has moved as requested
        pos['x'] -= 100
        microscope.move(pos) # moves microscope back to its starting position
        assert microscope.position == starting_pos
        if microscope.position==starting_pos:
            print('positioning is ok')
        else:
            raise ValueError(f'positioning is incorrect, expected {starting_pos} but received {microscope.position}')
            # return
#position()


# NEW AUTOFOCUS CODE
def autofocus_and_image(microscope, site_images_path, rows, cols, step_size): #, site_ids):
    os.makedirs(site_images_path, exist_ok=True)
    print('calibrating microscope stage position with camera...')
    microscope.calibrate_xy()    # THIS IS NOT WORKING IN Y?????????????????????????? figure out why this is ASAP.
    print('calibration complete')

    for row in range(rows):
        for column in range(cols):
            image_filename = f'Image_row:{row}_col:{column}.png' #_site id:{site_ids}.png'
            image_filepath = os.path.join(site_images_path, image_filename)

            if os.path.exists(image_filepath):
                print(f'skipping file {image_filepath} as it already exists')
                continue
            else:
                plt.imread(image_filepath)
                plt.title()
                plt.savefig(image_filepath, 'Image_row:{row}_col:{column}.png')

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
            print(f'image saved to {image_filepath}')
    print('scanning completed')


# DISPLYING & SAVING AN IMAGE
def image_looping(site_images_path): # do i need this?? - turn on when you want axes.
    # reads each image in site_file_path, adds a title and axes and saves to a new folder
    image_loop = True
    if image_loop:
        print(f"the number of images in site_images is {len(os.listdir(site_images_path))}")
        #for i, image in enumerate(site_images):
        for site_ids, site_file_name in enumerate(sorted(os.listdir(site_images_path)), start=1):
            site_file_path = os.path.join(site_images_path, site_file_name)
            image = plt.imread(site_file_path)
            print(f"working on image no.{site_ids} of {len(os.listdir(site_images_path))}")
            plt.imshow(np.array(image))
            plt.title(f'Image of Site No. {site_ids}')
            plt.xlabel('label this')
            plt.ylabel('label this')
            images_path_2 = os.path.join(site_images_2, f'Image_of_site {site_ids}.png')
            #plt.savefig(images_path_2)
            plt.show()
            plt.close()
            #print(f"image {site_ids} saved to {images_path_2}")
    #return image_loop

#image_loop = image_looping(site_images_path)


# FIX OR REMOVE THIS? A good feature identifying section may remove the need for colour identification at this stage? Can extract from spectra later.
def color_masking(site_images_path):
    color_masks = False
    masks = []
    if color_masks:
        #image_names = []
        dc = (0, 0, 0)  # change this, min hue, saturation, value min
        bc = (180, 255, 255)
        for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
            if image.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image)
                image_array = cv2.imread(image_path)
                print(f"READING IMAGE {site_ids}")
               # image_array = np.array(image)

                grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV) # converts to a hsv image
                hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)

                min_hue, max_hue = np.min(hue_channel), np.max(hue_channel)
                min_sat, max_sat = np.min(saturation_channel), np.max(saturation_channel)
                min_val, max_val = np.min(value_channel), np.max(value_channel)

                deepest_tuple = (min_hue, min_sat, min_val)
                brightest_tuple = (max_hue, max_sat, max_val)
                darkest_color = np.min(image_array)
                brightest_color = np.max(image_array)
                val_min = np.min(hsv_image)
                val_max = np.max(hsv_image)
                #print(f"{image}, site id {site_ids}: val max is {val_max} and val min is {val_min}")

                color_mask = cv2.inRange(hsv_image, np.array(deepest_tuple, dtype=np.uint8), np.array(brightest_tuple, dtype=np.uint8))
                #color_mask = np.uint8(color_mask)
                masks.append(color_mask)
                #print(f"the color mask is {color_mask}")
                #return color_mask
        return masks #, color_mask #, deepest_tuple, brightest_tuple

#masks, color_mask  = color_masking(site_images_path)  #  deepest_tuple, brightest_tuple

show_masks = False
if show_masks:
    plt.imshow(masks[0])
    plt.title(f'mask for first image')  # change the masks or remove them?
    plt.show()
    plt.close()



def canny_edge_detection(site_images_path, min_droplet_size=10, max_droplet_size=10000):
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
                edges = cv2.Canny(image_array, 100, 200)
                print(f'processing canny edges for image of site id {site_ids}')
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                tgt_features = []
                perimeters = []

                for contour in contours:
                    droplet_area = cv2.contourArea(contour)
                    if min_droplet_size < droplet_area < max_droplet_size:
                        print(f" droplet area {droplet_area} is ok")
                        M = cv2.moments(contour)  # this gives the contour area
                          # identifies if shape is a closed loop i.e. circle
                        if M['m00'] != 0:
                            x_centroid = int(M['m10'] / M['m00'])
                            y_centroid = int(M['m01'] / M['m00'])  # x and y centroids
                            perimeter = cv2.arcLength(contour, True)
                            # print(f"x centroid is {x_centroid} and the y centroid is {y_centroid}")
                            tgt_features.append((y_centroid, x_centroid))
                            perimeters.append(perimeter)
                            print(f'features for site {site_ids}: {tgt_features}')
                            print(f'perimeter for site{site_ids}: {perimeters}')

                        else:
                            print('contour has no area and has been skipped')

                all_tgt_features.append(tgt_features)
                all_perimeters.append(perimeters)

                fig, axs = plt.subplots(1,2, figsize=(10,5))
                axs[0].imshow(image_array, cmap='gray')
                axs[0].set_title(f'Original Image for Site {site_ids}')
                axs[1].imshow(edges, cmap='gray')
                axs[1].set_title(f'Edges for Site {site_ids}')
                plt.show()
                plt.savefig(f'canny_edges_site{site_ids}')
                #plt.close()

                for y,x in tgt_features:
                    axs[1].scatter(x, y, color='red', s=10)
                plt.show()
                plt.close()



    return all_tgt_features, all_perimeters

features, perimeters = canny_edge_detection(site_images_path, min_droplet_size=10, max_droplet_size=1e15)

# HOUGH
def hough_transforms(site_images_path, hough_output_path, dp=1, min_dist=5, param1=10, param2=10, min_radius = 10, max_radius=1e10):
    # param 1 defines how many edges are detected using the Canny edge detector (higher vals = fewer edges)
    # param 2 defines how many votes a circle must receive in order for it to be considered a valid circle (higher vals = a higher no. votes needed)
    hough = True
    if hough:
        for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
            if image.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image)
                color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                #gaussian_blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

                circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for x,y,r in circles[0,:]:
                        # draws the circles on the colour image
                        cv2.circle(color_image, (x,y), r, (0, 255, 0), 2)
                        cv2.circle(color_image, (x,y), 2, (0, 0, 255), 3)
                cv2.imshow(f'Detected Circles Using Hough Transforms for Site {site_ids}', color_image)
                #cv2.destroyAllWindows()
                cv2.imwrite(hough_output_path, color_image)
                print(f'hough transform image {site_ids} saved to {hough_output_path}')

hough_transforms(site)

# OLD SIZE MASK - REPLACED WITH CANNY EDGES AND HOUGH TRANSFORMS

# making a size mask i.e. if a droplet is within a certain size then do this - this works
# def size_masking(color_mask, min_droplet_size=10, max_droplet_size=10000):
#     size_mask = False
#     if size_mask:
#         #for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
#         contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             #tgt_contour = None
#         tgt_features = [] # storing features in image in a list
#         masked_image = np.zeros_like(color_mask, dtype=np.uint8)
#             #print(f"masked_image {site_ids} type is: {type(masked_image)}")
#             #print(f"masked_image {site_ids} shape is {masked_image.shape}")
#         for contour in contours:
#             droplet_area = cv2.contourArea(contour)
#               # what are the actual sizes???????
#                 #print(f"min size is {min_droplet_size} and max size is {max_droplet_size}")
#             if min_droplet_size < droplet_area < max_droplet_size:
#                 print(f" droplet area {droplet_area} is ok")
#                 M = cv2.moments(contour)  # this gives the contour area
#                 perimeter = cv2.arcLength(contour, True)  # identifies if shape is a closed loop i.e. circle
#                 if M['m00'] != 0:
#                     x_centroid = int(M['m10'] / M['m00'])
#                     y_centroid = int(M['m01'] / M['m00'])  # x and y centroids
#                     #print(f"x centroid is {x_centroid} and the y centroid is {y_centroid}")
#                     tgt_features.append((y_centroid, x_centroid))
#                     print(f' the target features are {tgt_features}')
#                 else:
#                     print('contour has no area and has been skipped')
#                     #tgt_contour = contour
#                     #cv2.drawContours(masked_image, [contour],-1, (0,255,0), 3)
#
#         return tgt_features
#


# i need to make some code to center a droplet that is interesting

def image_fun_sites(microscope, site_images_path, autofocus_and_image, tgt_features):
    image_sites = False
    if image_sites:
        for site_ids, site_coords in enumerate(tgt_features, start=1): # change this
            print(f"capturing site {site_ids} at coords {site_coords}")
            current_pos = microscope.position.copy()
            tgt_pos = current_pos.copy()
            tgt_pos['x'] += site_coords[0]
            tgt_pos['y'] += site_coords[1]
            microscope.move(tgt_pos)
            print(f"now moving microscope to {site_ids}")

            # now call image capturing defs
            autofocus_and_image(microscope, site_images_path, site_ids)
            tgt_features, masked_image = size_masking(color_mask=color_mask, min_droplet_size=0,max_droplet_size=1e15)

            print('image displayed')

    #return #autofocus_and_image  # may hash this out later



masked_contours = False
if masked_contours:
    plt.imshow(masked_image)
    print(f'the target features are {tgt_features}')
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title('masked image with contours shown')
    plt.show()
    image_fun_sites(microscope, site_images_path, autofocus_and_image, tgt_features)

# CHECK that ths works on Thursday 21/11/24. if it does then move onto extracting spectrum at each site and saving it.

# this works, I now need to fix the autofocus as it's still a little blurry. then I can gather some spectra
#Once that works, move on to image analysis of each spectrum site!

# SPECTRUM GATHERING CODE

def gather_spectra(spectrometer, spectra_path, features):
    spectra = True
    if spectra:
        os.makedirs(spectra_path, exist_ok=True)
        for site_ids, site_coords in enumerate(features, start=1):
            try:
                spectrum = spectrometer.spectrum()
                wavelengths = spectrum[0] # extracting wavelengths
                intensities = spectrum[1] # extracting intensities

                spectra_save_location = os.path.join(spectra_path, f'spectra of site{site_ids}.csv')  # saving as a csv file
                np.savetxt(spectra_save_location, np.column_stack((wavelengths,intensities)), delimiter=',', header='wavelength,intensity')
                print(f' spectrum {site_ids} saved to {spectra_save_location}')
            except Exception as e:
                print(f' error when saving spectrum {site_ids}, {e}')

def connect_to_spectrometer():
    devices = sb.list_devices()
    if devices:
        spectrometer = sb.Spectrometer(devices[0])
        print('Spectrometer is on')
        return spectrometer
    else:
        print("No spectrometer found")
        return None

def spectra_of_sites(microscope, spectrometer, spectra_path):
    spectra2 = False
    if spectra2:
        #tgt_features, masked_image = size_masking(color_mask=masks, min_droplet_size=0,max_droplet_size=1e15)
        features, perimeters = canny_edge_detection(site_images_path, min_droplet_size=10, max_droplet_size=1e15)
        for site_ids, site_coords in enumerate(features, start=1):

            current_pos = microscope.position.copy()
            tgt_pos = current_pos.copy()
            tgt_pos['x'] += site_coords[0]
            tgt_pos['y'] += site_coords[1]
            microscope.move(tgt_pos)
            #autofocus_and_image(microscope, site_images_path, site_ids)
            #print(f'image displayed of site {site_ids}')


            gather_spectra(spectrometer, spectra_path, features)
            print(f'spectra saved of site {site_ids}')

#spectra_of_sites(microscope, spectrometer, spectra_path)

#spectrometer.close()


def csv_to_png(csv_inputs, png_outputs):
    csv = True
    if csv:
        #if not os.path.exists(png_outputs):
        os.makedirs(png_outputs, exist_ok=True)
        for site_ids, filename in enumerate(os.listdir(csv_inputs), start=1):
            if filename.endswith('.csv'):
                try:
                    csv_filepath =  os.path.join(csv_inputs, filename)
                    png_filepath = os.path.join(png_outputs, filename.replace('.csv', '.png'))
                    numpy_data = np.loadtxt(csv_filepath, delimiter=',', skiprows=1)
                    wavelengths = numpy_data[:,0]
                    intensities = numpy_data[:,1]

                    # png plot
                    # plt.figure...
                    plt.plot(wavelengths, intensities)
                    plt.title(f'Spectrum of site {site_ids}')
                    plt.xlabel(f'Wavelength (nm)')
                    plt.ylabel(f'Intensity')   # potential units (Wm^-2 nm^-1)')
                    plt.savefig(png_filepath, format='png')
                    plt.show()
                    print(f"spectra displayed of site {site_ids}")
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
        gather_spectra(spectrometer, spectra_path, features)

#spectra_of_sites(microscope, spectrometer, spectra_path)
png_output_folder = 'C:/Users/shann/PycharmProjects/P4 Project/spectra_png'
csv_to_png(spectra_path, png_output_folder)



# check if this works 22/11/24. if so, then figure out a way to identify certain features in your spectra over the weekend!
# issues
# i need to skip images that already exist
# i need to fix the autofocus
# i need to improve the feature identifying section - potentially using canny edge detection to identify any features??? - complete


# SPECTRA ANALYSIS SECTION!!!
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_spectra(spectrum_file):
    try:
        df = pd.read_csv(spectrum_file, skiprows=1, names=["Wavelength", "Intensity"])
        df = df.astype({"Wavelength": float, "Intensity": float})
        wavelengths = df['Wavelength'].values
        spectrum = df['Intensity'].values
        return wavelengths, spectrum
    except Exception as e:
        raise RuntimeError(f"error loading spectrum data for file {spectrum_file}: {e}")

#wavelengths, spectrum = load_spectra(spectrum_file)

def compare_peaks(wavelengths, spectrum, known_lines=None, title="Spectrum"): # add site_ids in here
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, spectrum, label="Observed Spectrum", color='blue', linewidth=1.5)

    # if known_lines:
    #     for substance, line_wavelengths in known_lines.items():
    #         for wavelength in line_wavelengths:
    #             plt.axvline(x=wavelength, linestyle="--", linewidth=1, label=f"{substance} {wavelength} nm")

    if known_lines:
        #colors = cm.get_cmap('tab10', len(known_lines))
        #for idx, (substance, line_wavelengths) in enumerate(known_lines.items()):
            # color = colors(idx)
            # for wavelength in line_wavelengths:
            #     plt.axvline(x=wavelength,linestyle="--",linewidth=1,color=color,label=f"{substance} {wavelength} nm")

        colors = plt.cm.tab10(np.linspace(0, 1, len(known_lines)))
        for i, (substance, line_wavelengths) in enumerate(known_lines.items()):
            for j, wavelength in enumerate(line_wavelengths):
                plt.axvline(x=wavelength, linestyle="--", linewidth=1, color=colors[i])
            plt.axvline(x=line_wavelengths[0], linestyle="--", linewidth=1, color=colors[i], label=substance)


    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectrum with Detected Peaks for Site") # add site_ids here
    plt.legend()
    #plt.grid(alpha=0.5)
    plt.show()

def process_all_spectra(spectrum_path, known_lines):
    """
    Process all spectra files in a given folder.
    """
    for file_name in os.listdir(spectrum_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(spectrum_path, file_name)
            print(f"Processing file: {file_name}")

            try:
                wavelengths, spectrum = load_spectra(file_path)
                compare_peaks(wavelengths, spectrum, known_lines, title=f"Spectrum: {file_name}")
            except RuntimeError as e:
                print(e)


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

process_all_spectra(spectrum_path, known_lines)




