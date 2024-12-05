# HOLDING SPACE

ransac_path = 'C:/Users/shann/PycharmProjects/P4 Project/ransac/10j'
#ransac_path = 'C:/Users/shann/PycharmProjects/P4 Project/ransac/BJ03'
#ransac_path = 'C:/Users/shann/PycharmProjects/P4 Project/ransac/BJ06'
from skimage.measure import ransac, CircleModel

def ransac_circle_detection(site_images_path, ransac_path, canny_threshold1=10,canny_threshold2=30,residual_threshold=2,min_radius=1,max_radius=10000):
    ransac_code = False
    if ransac_code:
        tgt_features_ransac = []
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
                    tgt_features_ransac.append((center_y, center_x))
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
        return detected_ransac_circles, tgt_features_ransac

#detected_ransac_circles, tgt_features_ransac = ransac_circle_detection(site_images_path, ransac_path)


# # OLD SIZE MASK - REPLACED WITH CANNY EDGES AND HOUGH TRANSFORMS
#
# # making a size mask i.e. if a droplet is within a certain size then do this - this works
# # def size_masking(color_mask, min_droplet_size=10, max_droplet_size=10000):
# #     size_mask = False
# #     if size_mask:
# #         #for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
# #         contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #             #tgt_contour = None
# #         tgt_features = [] # storing features in image in a list
# #         masked_image = np.zeros_like(color_mask, dtype=np.uint8)
# #             #print(f"masked_image {site_ids} type is: {type(masked_image)}")
# #             #print(f"masked_image {site_ids} shape is {masked_image.shape}")
# #         for contour in contours:
# #             droplet_area = cv2.contourArea(contour)
# #               # what are the actual sizes???????
# #                 #print(f"min size is {min_droplet_size} and max size is {max_droplet_size}")
# #             if min_droplet_size < droplet_area < max_droplet_size:
# #                 print(f" droplet area {droplet_area} is ok")
# #                 M = cv2.moments(contour)  # this gives the contour area
# #                 perimeter = cv2.arcLength(contour, True)  # identifies if shape is a closed loop i.e. circle
# #                 if M['m00'] != 0:
# #                     x_centroid = int(M['m10'] / M['m00'])
# #                     y_centroid = int(M['m01'] / M['m00'])  # x and y centroids
# #                     #print(f"x centroid is {x_centroid} and the y centroid is {y_centroid}")
# #                     tgt_features.append((y_centroid, x_centroid))
# #                     print(f' the target features are {tgt_features}')
# #                 else:
# #                     print('contour has no area and has been skipped')
# #                     #tgt_contour = contour
# #                     #cv2.drawContours(masked_image, [contour],-1, (0,255,0), 3)
# #
# #         return tgt_features
# #



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

# # FIX OR REMOVE THIS? A good feature identifying section may remove the need for colour identification at this stage? Can extract from spectra later.
# def color_masking(site_images_path):
#     color_masks = False
#     masks = []
#     if color_masks:
#         #image_names = []
#         dc = (0, 0, 0)  # change this, min hue, saturation, value min
#         bc = (180, 255, 255)
#         for site_ids, image in enumerate(os.listdir(site_images_path), start=1):
#             if image.endswith(('.png', '.jpg')):
#                 image_path = os.path.join(site_images_path, image)
#                 image_array = cv2.imread(image_path)
#                 print(f"READING IMAGE {site_ids}")
#                # image_array = np.array(image)
#
#                 grayscale = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
#                 hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV) # converts to a hsv image
#                 hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
#
#                 min_hue, max_hue = np.min(hue_channel), np.max(hue_channel)
#                 min_sat, max_sat = np.min(saturation_channel), np.max(saturation_channel)
#                 min_val, max_val = np.min(value_channel), np.max(value_channel)
#
#                 deepest_tuple = (min_hue, min_sat, min_val)
#                 brightest_tuple = (max_hue, max_sat, max_val)
#                 darkest_color = np.min(image_array)
#                 brightest_color = np.max(image_array)
#                 val_min = np.min(hsv_image)
#                 val_max = np.max(hsv_image)
#                 #print(f"{image}, site id {site_ids}: val max is {val_max} and val min is {val_min}")
#
#                 color_mask = cv2.inRange(hsv_image, np.array(deepest_tuple, dtype=np.uint8), np.array(brightest_tuple, dtype=np.uint8))
#                 #color_mask = np.uint8(color_mask)
#                 masks.append(color_mask)
#                 #print(f"the color mask is {color_mask}")
#                 #return color_mask
#         return masks #, color_mask #, deepest_tuple, brightest_tuple
#
# #masks, color_mask  = color_masking(site_images_path)  #  deepest_tuple, brightest_tuple
#
# show_masks = False
# if show_masks:
#     plt.imshow(masks[0])
#     plt.title(f'mask for first image')  # change the masks or remove them?
#     plt.show()
#     plt.close()
#



# OLD AUTOFOCUS CODE
# def autofocus_and_image(microscope, site_images_path, rows, columns, step_size):
#     site_images = []
#     autofocus = False
#     if autofocus:
#         for site_ids, site_file_name in enumerate(sorted(os.listdir(site_images_path)), start=1):
#             try:
#                 ret = microscope.autofocus()
#                 #ret = microscope.laplacian_autofocus()
#                 if ret:
#                     print(f"autofocus was successful at {site_ids}") # with a value of {ret}")
#                     #image = microscope.grab_image()
#                     image = microscope.grab_image_array()
#
#                     if image is not None:
#                         image_path = os.path.join(site_images_path, f'Image_{site_ids}.png')
#                         plt.imsave(image_path, np.array(image))
#                         site_images.append(image)
#                         plt.imshow(image) # may remove this
#                         print(f"image of site {site_ids} appended to list & saved") # saves just the image
#
#                         if len(site_images) == 1:
#                             plt.imshow(image)
#                             plt.title(f"first image, site {site_ids}")
#                             #plt.axis('off')
#                             #plt.show()
#                             plt.close()
#                     else:
#                         print('image is none')
#                 else:
#                     print(f"autofocus was not successful at {site_ids}")
#             except Exception as e:
#                 print(f" error at site {site_ids}, {e}")
#     return site_images

#autofocus_and_image(microscope, site_images_path)
