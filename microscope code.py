

import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import openflexure_microscope_client as ofm_client
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import contour

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
# origin = "http://169.254.24.243:5000" # this is my microscope
# origin = "http://169.254.193.20:5000" # this is dr mehr's microscope

# Q: can I load spectra from images of a site?? or is it directly from the beam splitter??


# TURN THESE INTO DEFS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# POSITION CODE
check_position = True
#def check_position_code:
if check_position:
    pos = microscope.position
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
        raise ValueError('positioning is wrong')
    #return



# AUTOFOCUS CODE
def autofocus_and_image(microscope, site_images_path, site_ids):
    site_images = []
    autofocus = True
    if autofocus:
        try:
            ret = microscope.autofocus()
            if ret:
                print(f"autofocus was successful at {site_ids} with a value of {ret}")
                image = microscope.grab_image()
                if image is not None:
                    site_images.append(image)
                    image_path = os.path.join(site_images_path, f'Image_{i + 1}.png')
                    plt.imsave(image_path, np.array(image))
                    print(f"image no {i+1} appended to list & saved") # saves just the image
                else:
                    print('image is none')
            else:
                print(f"autofocus was not successful at {site_ids}")
        except Exception as e:
            print(f" error at site {site_ids}, {e}")
    return site_images



# DISPLYING & SAVING AN IMAGE
def image_looping(site_images_path): # do i need this?? - turn on when you want axes.
    # reads each image in site_file_path, adds a title and axes and saves to a new folder
    image_loop = False
    if image_loop:
        print(f"the number of images in site_images is {len(site_images_path)}")
        #for i, image in enumerate(site_images):
        for i, site_file_name in enumerate(sorted(os.listdir(site_images_path))):
            site_file_path = os.path.join(site_images_path, site_file_name)
            image = plt.imread(site_file_path)
            print(f"working on image no.{i} of {len(site_images_path)}")
            plt.imshow(np.array(image))
            plt.title(f'Image of Site No. {i + 1}')
            plt.xlabel('label this')
            plt.ylabel('label this')
            images_path_2 = os.path.join(site_images_2, f'Image_of_site {i + 1}.png')
            plt.savefig(images_path_2) # i need to save this image with axes and a title - done
            plt.show()
            print(f"image {i+1} saved to {images_path_2}")
    return image_loop



#making a color mask
def color_masking(image, site_images_path):
    color_mask = True
    if color_mask:
        for i, image in enumerate(os.listdir(site_images_path)):
            image_array = np.array(image)
            grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            color_filter = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            #print(f"the color filter is {color_filter}")
            # saturation_min = ...
            # saturation_max = ...
            val_min = np.min(color_filter)
            val_max = np.max(color_filter) # defining 256 val
            print(f"val max is {val_max} and val min is {val_min}")
            deepest_color = np.array([val_min])
            brightest_color = np.array([val_max])
            color_mask = cv2.inRange(color_filter, deepest_color, brightest_color)
            #print(f"the color mask is {color_mask}")
    return color_mask

# choose which size mask to use!

# making a size mask i.e. if a droplet is within a certain size then do this - this works
def size_masking(color_masking):
    size_mask = True
    if size_mask:
        contours, _ = cv2.findContours(color_masking, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tgt_contour = None
        for contour in contours:
            droplet_area = cv2.contourArea(contour)
            min_droplet_size = 100
            max_droplet_size = 500  # what are the actual sizes???????
            print(f"min size is {min_droplet_size} and max size is {max_droplet_size}")
            if min_droplet_size < droplet_area < max_droplet_size:
                print(f" droplet area {droplet_area} is ok")
                target_contour = contour
                break
    return target_contour, contours

target_contour = size_masking(color_masking)[0]
contours = size_masking(color_masking)[1]

# i need to make some code to center a droplet that is interesting

def find_sites(contours):
    fun_sites = []
    for contour in contours:
        M = cv2.moments(contour)  # this gives the contour area
        perimeter = cv2.arcLength(contour, True) # identifies if shape is a closed loop i.e. circle
        if M['m00'] != 0:
            x_centroid = int(M['m10'] / M['m00'])
            y_centroid = int(M['m01'] / M['m00']) # x and y centroids
            fun_sites.append((y_centroid, x_centroid))

    return fun_sites

def image_fun_sites(microscope, site_images_path, autofocus_and_image, find_sites, site_ids):
    for site_ids, site_coords in enumerate(find_sites, start=1):
        print(f"capturing site{site_ids} at coords {site_coords}")
        current_pos = microscope.position.copy()
        tgt_pos = current_pos.copy().copy()
        tgt_pos['x'] += site_coords[0]
        tgt_pos['y'] += site_coords[1]
        microscope.move(tgt_pos)
        print(f"now moving microscope to {site_ids}")

        # now call image capturing defs
        autofocus_and_image(microscope, site_images_path, site_ids)
        print('image displayed')

    return autofocus_and_image  # may hash this out later

image_fun_sites(microscope, target_contour, site_images_path, autofocus_and_image)

# CHECK that ths works on Monday 18/11/24. if it does then move onto extracting spectrum at each site and saving it.
#Once that works, move on to image analysis of each spectrum site!



# class spectrometer_control:
#     def __init__(self):
#         # connects to spectrometer
#         print("Connected to the spectrometer")
#
#     def spectrum(self):
#         # gather spectrum from the site chosen here ADD CODE 08/11/24
#         print("gathering spectrum")
#         return # return spectral data here
#

#             # Acquire spectrum
#             spectrum = self.spectrometer.acquire_spectrum()
#             results[(x, y, z)] = spectrum
#             print(f" spectrum for ste {i + 1} collected.")
#
#         return results
#
#
# # ...
# microscope = microscope_control()
# spectrometer = spectrometer_control()
#
# # automation
# automation = automation_controller(microscope, spectrometer)
#
# # interesting sites to visit (placeholder coords)
# # #( i want to develop this so that interesting sites are found rather than me manually choosing them)
# automation.add_site(10, 20, z) # placeholdersss, add z position
# automation.add_site(30, 40, z)
# automation.add_site(50, 60, z)
#
# # runs automation
# spectra_data = automation.visit_sites_and_gather_spectra()
#
# # prints results i.e. the site and the data
# for (x, y, z), spectrum in spectra_data.items():
#     print(f"Site ({x}, {y}, {z}) Spectrum Data: {spectrum}")
