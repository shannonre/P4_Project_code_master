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

rows = 3
cols = 3
step_size = 250

#Goes through 3 rows and 3 cols for a total of 9 images with a step size of x or y = 1000 per row. Also performs autofocus at each point.


# POSITION CHECK
def position():
    check_position = False
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
    imaging = False
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
def image_looping(site_images_path): # do i need this?? - turn on when you want axes.
    # reads each image in site_file_path, adds a title and axes and saves to a new folder
    image_loop = False
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
            #plt.show()
            plt.close()
            #print(f"image {site_ids} saved to {images_path_2}")
    #return image_loop

#image_loop = image_looping(site_images_path)

# FEATURE IDENTIFICATION

canny_path = 'C:/Users/shann/PycharmProjects/P4 Project/canny'

def canny_edge_detection(site_images_path, canny_path, min_droplet_size=1, max_droplet_size=10000):
    canny1 = False
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
                axs[1].set_title(f'Canny Edge Detection for Site {site_ids}')

                #plt.close()
                #for y,x in tgt_features:
                    #axs[1].scatter(x, y, color='red', s=10)
                #plt.show()
                plt.savefig(os.path.join(canny_path, f'canny_edges_site{site_ids}'))
                plt.show()
                # plt.close()



    return all_tgt_features, all_perimeters

features, perimeters = canny_edge_detection(site_images_path, canny_path, min_droplet_size=10, max_droplet_size=1e15)

# HOUGH
hough_path = 'C:/Users/shann/PycharmProjects/P4 Project/hough'
def hough_transforms(site_images_path, hough_path, dp=1, min_dist=5, param1=10000, param2=10000000, min_radius = 120, max_radius=1000):
    # param 1 defines how many edges are detected using the Canny edge detector (higher vals = fewer edges)
    # param 2 defines how many votes a circle must receive in order for it to be considered a valid circle (higher vals = a higher no. votes needed)
    hough = True
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

                circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    detected_circles[site_ids] = circles[0, :]
                    for x,y,r in circles[0,:]:
                        # draws the circles on the colour image
                        cv2.circle(color_image, (x,y), r, (255, 0, 0), 1)
                        #cv2.circle(color_image, (x,y), 2, (0, 0, 255), 3)
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
        return detected_circles

detected_circles = hough_transforms(site_images_path, hough_path,min_dist=15, param1=10, param2=100, min_radius=10, max_radius=10000)
print("Detected circles:", detected_circles)


ransac_path = 'C:/Users/shann/PycharmProjects/P4 Project/ransac'
from skimage.measure import ransac, CircleModel

def ransac_circle_detection(site_images_path, ransac_path, canny_threshold1=10,canny_threshold2=30,residual_threshold=2,min_radius=1,max_radius=10000):
    ransac_code = False
    if ransac_code:
        if not os.path.exists(ransac_path):
            os.makedirs(ransac_path)
        detected_ransac_circles = {}
        for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
            if image.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
                y_coords, x_coords = np.nonzero(edges)
                edge_points = np.column_stack((x_coords, y_coords))
                print(f"No. edge points detected for site {site_ids}: {len(edge_points)}")


                model_class = CircleModel
                try:
                    ransac_model, inliers = ransac(edge_points,model_class, min_samples=3,residual_threshold=residual_threshold,max_trials=100000)
                except ValueError as e:
                    print(f"RANSAC failed for site {site_ids}: {e}")
                    continue

                if ransac_model is not None and ransac_model.params is not None:
                    center_x, center_y, radius = ransac_model.params
                    print(f"Detected circle for site {site_ids}: Center=({center_x:.2f}, {center_y:.2f}), Radius={radius:.2f}")

                    if not (min_radius <= radius <= max_radius):
                        print(f"Detected radius is out of the specified bounds for site {site_ids}.")
                        continue
                    cv2.circle(image, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 2)
                    cv2.circle(image, (int(center_x), int(center_y)), 2, (0, 0, 255), 3)

                    detected_ransac_circles[site_ids] = (center_x, center_y, radius)
                else:
                    print(f"No circles detected for site {site_ids}.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(8, 8))
                plt.imshow(image_rgb)
                plt.title(f"Detected Circles using RANSAC for Site {site_ids}")
                plt.axis("off")
                plt.savefig(os.path.join(ransac_path, f'ransac detection {site_ids}'))
                plt.show()

                # place some saving code here

                # Return the model parameters if needed
        return detected_ransac_circles

ransac_circle_detection(site_images_path, ransac_path)

sift_results_path = "C:/Users/shann/PycharmProjects/P4 Project/sift_results"
def sift_feature_detection(site_images_path, sift_results_path):
    sift = True
    if sift:
        tgt_features = []
        if not os.path.exists(sift_results_path):
            os.makedirs(sift_results_path)

        sift = cv2.SIFT_create()

        for i, image_name in enumerate(os.listdir(site_images_path), start=1):
            if image_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(site_images_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                keypoints, descriptors = sift.detectAndCompute(gray, None)

                print(f"Site {i}: detected {len(keypoints)} keypoints")
                for kp in keypoints:
                    x_coord, y_coord = kp.pt
                    tgt_features.append((int(y_coord), int(x_coord)))
                    print(f" target feature at (y={y_coord}, x={x_coord})")

                # Draw keypoints on the image
                keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                keypoint_image_rgb = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2RGB)

                plt.figure(figsize=(10, 10))
                plt.imshow(keypoint_image_rgb)
                plt.title(f"SIFT Features for Site {i}")
                plt.axis("off")
                plt.savefig(os.path.join(sift_results_path, f"sift_features_site_{i}.png"))
                plt.show()

                descriptors_path = os.path.join(sift_results_path, f"descriptors_site_{i}.npy")
                np.save(descriptors_path, descriptors)
        print(f' total no. target features = {tgt_features}')
        return tgt_features


tgt_features = sift_feature_detection(site_images_path, sift_results_path)




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

# SPECTRUM GATHERING CODE
# change features depending on canny/hough/ransac

def gather_spectra(spectrometer, spectra_path, features):
    spectra = False
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
    csv = False
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
# i need to fix the autofocus






