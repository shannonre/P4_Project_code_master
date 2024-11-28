# HOLDING SPACE







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
