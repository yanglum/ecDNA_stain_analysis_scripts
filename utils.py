#!/usr/bin/env python
# coding: utf-8

# ### change log
# 24-08-20: add connected component analysis to count # of CCs for each stain within each ROI

# python 3.8.5, skimage 0.18.1
# env = cv2
# uses ROI_selector_config.yaml
# IMPORTANT: does not handle double digit channel numbers

import os
import pickle
from skimage import io, img_as_ubyte, measure, draw, morphology
import numpy as np
import pandas as pd
import cv2
from skimage.filters import threshold_multiotsu, threshold_otsu

def sort_channel(inpath, channels):
	for channel in channels:
		if not os.path.isdir(os.path.join(inpath, channel)):
			os.mkdir(os.path.join(inpath, channel))
	   
	for file in os.listdir(inpath):
		if not os.path.isdir(os.path.join(inpath, file)):
			index = file.find("channel")
			chno = int(file[index+len("channel")])
			os.rename(os.path.join(inpath, file), os.path.join(inpath, channels[chno], file))
			
def combine_channel(inpath, channels, colors):
	outputadd = "_".join(channels)

	color_code = []
	for color in colors:
		if color == "blue":
			color_code.append(2)
		elif color == "red":
			color_code.append(0)
		elif color == "green":
			color_code.append(1)

	master_dict = {} # dictionary of channels, each of which is a dictionary
	for channel in channels:
		master_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_dict[channel][file.split("_")[0]] = file

	# get list of image#s
	files = os.listdir(os.path.join(inpath, channels[0]))
	images = []
	extension = files[0][-4:] # get extension of files (.png or .tif, etc)
	test = io.imread(os.path.join(inpath, channels[0], files[0])) # for obtaining size of images
	for file in files:
		if file.endswith('.tif') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.TIF') or file.endswith('.PNG') or file.endswith('.JPG'):
			images.append(file.split('_')[0])
		
	for image in images:
		image_dict = {}
		for i, channel in enumerate(master_dict.keys()):
			try:
				image_dict[color_code[i]] = io.imread(os.path.join(inpath, channel, master_dict[channel][image]))
			except:
				pass
		combined_image = np.zeros((test.shape[0], test.shape[1], 3), dtype=np.uint8)
		for key in image_dict:
			combined_image[:,:,key] = img_as_ubyte(image_dict[key])
		io.imsave(os.path.join(inpath, image+"_"+outputadd+extension), combined_image, check_contrast=False)
		print('.', end='', flush=True)	 

def stain_segmentation(inpath, channels, high_thresh, low_thresh, combine_masks, close_holes):
	channel_paths = {}
	
	if len(combine_masks)==2:
		imagenumbers = []
		to_combine = {}
		combined_mask_dir = os.path.join(inpath, combine_masks[0]+'_'+combine_masks[1]+'_labels')
		if not os.path.isdir(combined_mask_dir):
			os.mkdir(combined_mask_dir)
			
	for channel in channels:
		channel_paths[channel] = os.path.join(inpath, channel)
	for channel in channels:
		if channel in combine_masks:
			to_combine[channel] = []
		print('\n\tworking on '+channel, end='', flush=True)
		label_dir = os.path.join(inpath, channel, 'labels')
		if not os.path.isdir(label_dir):
			os.mkdir(label_dir)
		if channel in high_thresh:
			segment_factor = 0
		elif channel in low_thresh:
			segment_factor = 1
		img_list = os.listdir(channel_paths[channel])
		for file in img_list:
			if file.endswith('.png') or file.endswith('.tif') or file.endswith('.jpg'):
				image = cv2.imread(os.path.join(channel_paths[channel], file), cv2.IMREAD_GRAYSCALE)
				try:
					thresholds = threshold_multiotsu(image)
					segments = np.digitize(image, bins=thresholds)
					segmented_image = segments>=np.max(segments)-segment_factor
				except ValueError: # if there are only 2 values in the image, cannot threshold into 3 classes
					thresholds = threshold_otsu(image)
					segmented_image = image > thresholds
				if channel in close_holes:
					segmented_image = morphology.area_closing(segmented_image, area_threshold=2977, connectivity=1, parent=None, tree_traverser=None)
				segmented_image = segmented_image.astype('uint8')*255
				cv2.imwrite(os.path.join(label_dir, file), segmented_image)
				
				if channel in combine_masks:
					imagenumber = file.split('_')[0]
					if imagenumber not in imagenumbers:
						imagenumbers.append(imagenumber)
					to_combine[channel].append(segmented_image.copy())
					
				print('.', end='', flush=True)
				
	if len(combine_masks)==2: 
		print('\ncombining masks: '+combine_masks[0]+' and '+combine_masks[1], end='', flush=True)

		for i, imagenumber in enumerate(imagenumbers):
			inter_mask = to_combine[combine_masks[0]][i] | to_combine[combine_masks[1]][i]
			inter_mask = morphology.area_closing(inter_mask, area_threshold=2977, connectivity=1, parent=None, tree_traverser=None)
			cv2.imwrite(os.path.join(combined_mask_dir, imagenumber+'_combined_mask.png'), inter_mask)
			print('.', end='', flush=True)

def expand(selection, radius):
	# expands a boolean mask "True" area by given radius in x and y dimensions
	cop = np.copy(selection)
	for x in range(-radius,radius+1):
		for y in range(-radius,radius+1):
			if (y==0 and x==0) or (x**2 + y**2 > radius**2):
				continue
			shift = np.roll(np.roll(selection, y, axis = 0), x, axis = 1)
			cop += shift
	return cop
	
def analyze_untethering(inpath, channels, chrDNA_channel, ecDNA_channel, no_nuclei, expand_chrDNA_pixel, ROIs, combine_masks):
	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	# make dictionaries

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	for channel in channels:
		master_path_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_path_dict[channel][file.split("_")[0]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file
				
	if len(combine_masks)==2:
		combined_mask_channel = combine_masks[0]+'_'+combine_masks[1]+'_labels'
		master_path_dict[combined_mask_channel] = {}
		for file in os.listdir(os.path.join(inpath, combined_mask_channel)):
			master_path_dict[combined_mask_channel][file.split("_")[0]] = file
				
	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
		'chrDNA_pixel_count': [], # pixel count of the large connected component of chrDNA_channel
		'chrDNA_total_intensity': [], # total intensity of the large connected component of chrDNA_channel
		'chrDNA_ecDNA_overlap_pixel_count': [],
	}

	channel_list_dict = {}
	for channel in channels:
		channel_list_dict[channel+'_pixel_count'] = []
		channel_list_dict[channel+'_total_intensity'] = []
		channel_list_dict[channel+'_connected_component_count'] = [] # 24_08_20						

	# test
	test_path = os.path.join(inpath, channels[0], 'labels')
	test_file = os.listdir(test_path)[0]
	test_mask = io.imread(os.path.join(test_path, test_file))

	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'polygon':
						rr, cc = draw.polygon(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'rectangle':
						rr, cc = draw.rectangle(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped', end='', flush=True)
					continue

				dna_masks_dict = {'chrDNA':[], 'ecDNA':[]}

				## for each channel
				for channel in channels:
				
					# load image and mask (for given channel and imageno)
					img_path = os.path.join(inpath, channel, master_path_dict[channel][imageno])
					mask_path = os.path.join(inpath, channel, 'labels', master_path_dict[channel][imageno])
					image = io.imread(img_path)
					mask = io.imread(mask_path)
					mask = mask==255

					# find intersection of channel mask and roi mask
					inter_mask = roi_mask & mask

					pixel_count = np.count_nonzero(inter_mask)
					total_intensity = sum(image[inter_mask])
					
					# 24_08_31	
					labeled_image = measure.label(inter_mask, connectivity=2)
					props = measure.regionprops(labeled_image)
					CCcount = 0
					for prop in props:
						if prop.area >= 3: # only count connected components with size greater than 5 pixels
							CCcount += 1				 

					# append channel lists
					channel_list_dict[channel+'_pixel_count'].append(pixel_count)
					channel_list_dict[channel+'_total_intensity'].append(total_intensity)
					channel_list_dict[channel+'_connected_component_count'].append(CCcount) # 24_08_31				  

					if channel == chrDNA_channel:
						if len(combine_masks)==2:
							mask_path = os.path.join(inpath, combined_mask_channel, master_path_dict[combined_mask_channel][imageno])
							mask = io.imread(mask_path)
							mask = mask==255
							inter_mask = roi_mask & mask
						try:
							# find and keep only the largest connected component (chrDNA)
							labeled_mask, num_components = measure.label(inter_mask, connectivity=2, return_num=True)
							props = measure.regionprops(labeled_mask)
							
							areas = [prop.area for prop in props]
							# Finding the indices that would sort the areas in descending order
							sorted_indices = np.argsort(-np.array(areas))
							# Create a new binary mask with only the largest connected component
							largest_component_mask = np.zeros_like(inter_mask, dtype=bool)
							if no_nuclei==1: # use for interphase, prophase, metaphase
								largest_component_mask[labeled_mask == sorted_indices[0] + 1] = True # add 1 because skimage.measure.label starts at 1, but np.array starts at 0
							elif no_nuclei==2: # use for anaphase, telophase
								# Find the indices of the largest 2 connected components based on the area
								if len(sorted_indices) == 1: # if only 1 nuclei object detected
									largest_component_mask[labeled_mask == sorted_indices[0] + 1] = True
								elif np.array(areas)[sorted_indices][0] / np.array(areas)[sorted_indices][1] > 2.8:
									# if the largest area is more than 2.8x bigger than 2nd largest, then only keep the largest
									# since the largest area likely contains both nuclei, and the 2nd largest is likely micronuclei or debris
									largest_component_mask[labeled_mask == sorted_indices[0] + 1] = True
								else:
									largest_component_mask[(labeled_mask == sorted_indices[0] + 1) | (labeled_mask == sorted_indices[1] + 1) ] = True
							else:
								print("can't do more than 2 nuclei yet!")								 
							# expand the "True" area of the mask by n pixels:
							largest_component_mask = expand(largest_component_mask, expand_chrDNA_pixel)
							chrDNA_pixel_count = np.count_nonzero(largest_component_mask)
							chrDNA_total_intensity = sum(image[largest_component_mask])

							# append lists
							analysis_dict['chrDNA_pixel_count'].append(chrDNA_pixel_count)
							analysis_dict['chrDNA_total_intensity'].append(chrDNA_total_intensity)

							dna_masks_dict['chrDNA'].append(largest_component_mask)
						except:
							analysis_dict['chrDNA_pixel_count'].append('error')
							analysis_dict['chrDNA_total_intensity'].append('error')

							dna_masks_dict['chrDNA'].append(inter_mask)
							print('error detected: check input ROIs and masks', end='', flush=True)
					elif channel == ecDNA_channel:
						dna_masks_dict['ecDNA'].append(inter_mask)


				# outside of channels loop
				# find intersection of chrDNA mask and ecDNA mask
				tethered_mask = dna_masks_dict['chrDNA'][0] & dna_masks_dict['ecDNA'][0]
				tethered_pixel_count = np.count_nonzero(tethered_mask)

				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
				analysis_dict['chrDNA_ecDNA_overlap_pixel_count'].append(tethered_pixel_count)
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(channel_list_dict)
	df = pd.concat([series1, series2], axis=1)
	df['percent_tethered'] = df['chrDNA_ecDNA_overlap_pixel_count']/df['mycfish_pixel_count']
	df['percent_untethered'] = 1-df['percent_tethered']

	df.to_csv(os.path.join(inpath, 'untethering_analysis.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)

def analyze_micronuclei(inpath, channels, chrDNA_channel, ecDNA_channel, no_nuclei, expand_chrDNA_pixel, ROIs, combine_masks): 
	mnseg_path = os.path.join(inpath, 'labels')

	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	# make dictionaries

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	master_path_dict['mnseg_mask'] = {}
	for channel in channels:
		master_path_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_path_dict[channel][file.split("_")[0]] = file
	for file in os.listdir(mnseg_path):
		if 'updated_' in file: # only use updated masks
			master_path_dict['mnseg_mask'][file.split("_")[1]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file
	if len(combine_masks)==2:
		try:
			combined_mask_channel = combine_masks[0]+'_'+combine_masks[1]+'_labels'
			master_path_dict[combined_mask_channel] = {}
			for file in os.listdir(os.path.join(inpath, combined_mask_channel)):
				master_path_dict[combined_mask_channel][file.split("_")[0]] = file
		except FileNotFoundError:
			combined_mask_channel = mnseg_path
			master_path_dict[combined_mask_channel] = {}
			for file in os.listdir(os.path.join(inpath, combined_mask_channel)):
				master_path_dict[combined_mask_channel][file.split("_")[0]] = file
			
	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
		'mnseg_nuc_size': [],
		'mnseg_mn_size': [],
		'mnseg_mn_count': [],
		'chrDNA_nuc_pixel_count': [], # pixel count of the large connected component of chrDNA_channel
		'chrDNA_mn_pixel_count': [], # total intensity of the large connected component of chrDNA_channel
		'chrDNA_ecDNA_nuc_overlap_pixel_count': [],
		'chrDNA_ecDNA_mn_overlap_pixel_count': [],
		'chrDNA_only_nuc_pixel_count': [],
		'chrDNA_only_mn_pixel_count': [],
		'ecDNA_only_nuc_pixel_count': [],
		'ecDNA_only_mn_pixel_count': [],
	}

	channel_list_dict = {}
	for channel in channels:
		channel_list_dict[channel+'_pixel_count'] = []
		channel_list_dict[channel+'_total_intensity'] = []
		channel_list_dict[channel+'_connected_component_count'] = [] # 24_08_20
		channel_list_dict[channel+'_nuc_pixel_count'] = []
		channel_list_dict[channel+'_nuc_total_intensity'] = []
		channel_list_dict[channel+'_nuc_connected_component_count'] = [] # 24_08_20
		channel_list_dict[channel+'_mn_pixel_count'] = []
		channel_list_dict[channel+'_mn_total_intensity'] = []
		channel_list_dict[channel+'_mn_connected_component_count'] = [] # 24_08_20

	# test
	test_path = os.path.join(inpath, channels[0], 'labels')
	test_list = os.listdir(test_path)
	test_mask = io.imread(os.path.join(test_path, test_list[0]))

	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'polygon':
						rr, cc = draw.polygon(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'rectangle':
						rr, cc = draw.rectangle(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped due to unknown ROI shape', end='', flush=True)
					continue
				
				# mnseg
				try:
					mnseg_mask = io.imread(os.path.join(mnseg_path, master_path_dict['mnseg_mask'][imageno]))
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped due to lack of mask', end='', flush=True)
					continue
				nuc_mask = np.zeros(mnseg_mask.shape[:2], dtype=bool)
				mn_mask = np.zeros(mnseg_mask.shape[:2], dtype=bool)
				#nuc_mask[np.all(mnseg_mask==[255, 255, 153], axis=2) | np.all(mnseg_mask==[127, 201, 127], axis=2)] = True # this is if mnseg mask is used
				nuc_mask[np.all(mnseg_mask==[255, 255, 255], axis=2)] = True
				mn_mask[np.all(mnseg_mask==[240, 2, 127], axis=2)] = True
				
				nuc_roi_mask = roi_mask & nuc_mask
				mn_roi_mask = roi_mask & mn_mask
				nucpixel_count = np.count_nonzero(nuc_roi_mask)
				mnpixel_count = np.count_nonzero(mn_roi_mask)
				labeled_mask, mn_count = measure.label(mn_roi_mask, connectivity=2, return_num=True)
				
				analysis_dict['mnseg_nuc_size'].append(nucpixel_count)
				analysis_dict['mnseg_mn_size'].append(mnpixel_count)
				analysis_dict['mnseg_mn_count'].append(mn_count)
				
				
				dna_masks_dict = {'chrDNA_nuc':[], 'chrDNA_mn':[], 'ecDNA_nuc':[], 'ecDNA_mn':[]}

				## for each channel
				for channel in channels:

					# load image and mask (for given channel and imageno)
					img_path = os.path.join(inpath, channel, master_path_dict[channel][imageno])
					mask_path = os.path.join(inpath, channel, 'labels', master_path_dict[channel][imageno])
					image = io.imread(img_path)
					mask = io.imread(mask_path)
					mask = mask==255

					# find intersection of channel mask and roi mask
					inter_mask = roi_mask & mask
					pixel_count = np.count_nonzero(inter_mask)
					total_intensity = sum(image[inter_mask])
					# 24_08_31	
					labeled_image = measure.label(inter_mask, connectivity=2)
					props = measure.regionprops(labeled_image)
					CCcount = 0
					for prop in props:
						if prop.area >= 3: # only count connected components with size greater than 5 pixels
							CCcount += 1	
					
					# find intersection of channel mask and roi mask and mnseg_mask
					nucinter_mask = inter_mask & nuc_mask
					nuc_channel_pixel_count = np.count_nonzero(nucinter_mask)
					nuc_channel_total_intensity = sum(image[nucinter_mask])
					
					# 24_08_31	
					nuclabeled_image = measure.label(nucinter_mask, connectivity=2)
					nucprops = measure.regionprops(nuclabeled_image)
					nucCCcount = 0
					for prop in nucprops:
						if prop.area >= 3: # only count connected components with size greater than 5 pixels
							nucCCcount += 1	
					
					mninter_mask = inter_mask & mn_mask
					mn_channel_pixel_count = np.count_nonzero(mninter_mask)
					mn_channel_total_intensity = sum(image[mninter_mask])
					
					# 24_08_31	
					mnlabeled_image = measure.label(mninter_mask, connectivity=2)
					mnprops = measure.regionprops(mnlabeled_image)
					mnCCcount = 0
					for prop in mnprops:
						if prop.area >= 3: # only count connected components with size greater than 5 pixels
							mnCCcount += 1					 

					# append channel lists
					channel_list_dict[channel+'_pixel_count'].append(pixel_count)
					channel_list_dict[channel+'_total_intensity'].append(total_intensity)	
					channel_list_dict[channel+'_connected_component_count'].append(CCcount) # 24_08_31				 
					channel_list_dict[channel+'_nuc_pixel_count'].append(nuc_channel_pixel_count)
					channel_list_dict[channel+'_nuc_total_intensity'].append(nuc_channel_total_intensity)
					channel_list_dict[channel+'_nuc_connected_component_count'].append(nucCCcount) # 24_08_31
					channel_list_dict[channel+'_mn_pixel_count'].append(mn_channel_pixel_count)
					channel_list_dict[channel+'_mn_total_intensity'].append(mn_channel_total_intensity)
					channel_list_dict[channel+'_mn_connected_component_count'].append(mnCCcount) # 24_08_31					

					if channel == chrDNA_channel:
						if len(combine_masks)==2:
							mask_path = os.path.join(inpath, combined_mask_channel, master_path_dict[combined_mask_channel][imageno])
							mask = io.imread(mask_path)
							mask = mask==255
							inter_mask = roi_mask & mask
						try:
							nucinter_mask = inter_mask & nuc_mask
							chrDNA_nuc_pixel_count = np.count_nonzero(nucinter_mask)	  
							mninter_mask = inter_mask & mn_mask
							chrDNA_mn_pixel_count = np.count_nonzero(mninter_mask)
							dna_masks_dict['chrDNA_nuc'].append(nucinter_mask)
							dna_masks_dict['chrDNA_mn'].append(mninter_mask)
							# append lists
							analysis_dict['chrDNA_nuc_pixel_count'].append(chrDNA_nuc_pixel_count)
							analysis_dict['chrDNA_mn_pixel_count'].append(chrDNA_mn_pixel_count)

						except:
							analysis_dict['chrDNA_nuc_pixel_count'].append('error')
							analysis_dict['chrDNA_mn_pixel_count'].append('error')
							
							print('error detected: check input ROIs and masks', end='', flush=True)
					elif channel == ecDNA_channel:
							dna_masks_dict['ecDNA_nuc'].append(nucinter_mask)
							dna_masks_dict['ecDNA_mn'].append(mninter_mask)
						
				# outside of channels loop
				# find intersection of chrDNA mask and ecDNA mask
				nuc_overlap_mask = dna_masks_dict['chrDNA_nuc'][0] & dna_masks_dict['ecDNA_nuc'][0]
				nuc_overlap_pixel_count = np.count_nonzero(nuc_overlap_mask)
				mn_overlap_mask = dna_masks_dict['chrDNA_mn'][0] & dna_masks_dict['ecDNA_mn'][0]
				mn_overlap_pixel_count = np.count_nonzero(mn_overlap_mask)
				
				# find chr mask only (not ec mask)
				nuc_chronly_mask = dna_masks_dict['chrDNA_nuc'][0] & ~dna_masks_dict['ecDNA_nuc'][0]
				nuc_chronly_pixel_count = np.count_nonzero(nuc_chronly_mask)
				mn_chronly_mask = dna_masks_dict['chrDNA_mn'][0] & ~dna_masks_dict['ecDNA_mn'][0]
				mn_chronly_pixel_count = np.count_nonzero(mn_chronly_mask)
				
				# find ec mask only (not chr mask)
				nuc_econly_mask = dna_masks_dict['ecDNA_nuc'][0] & ~dna_masks_dict['chrDNA_nuc'][0]
				nuc_econly_pixel_count = np.count_nonzero(nuc_econly_mask)
				mn_econly_mask = dna_masks_dict['ecDNA_mn'][0] & ~dna_masks_dict['chrDNA_mn'][0]
				mn_econly_pixel_count = np.count_nonzero(mn_econly_mask)
				
				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
				analysis_dict['chrDNA_ecDNA_nuc_overlap_pixel_count'].append(nuc_overlap_pixel_count)
				analysis_dict['chrDNA_ecDNA_mn_overlap_pixel_count'].append(mn_overlap_pixel_count)
				analysis_dict['chrDNA_only_nuc_pixel_count'].append(nuc_chronly_pixel_count)
				analysis_dict['chrDNA_only_mn_pixel_count'].append(mn_chronly_pixel_count)
				analysis_dict['ecDNA_only_nuc_pixel_count'].append(nuc_econly_pixel_count)
				analysis_dict['ecDNA_only_mn_pixel_count'].append(mn_econly_pixel_count)
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(channel_list_dict)
	df = pd.concat([series1, series2], axis=1)

	df.to_csv(os.path.join(inpath, 'micronuclei_analysis.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)
	
def line_intensity_analysis(inpath, channels, ROIs):
	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	# make dictionaries

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	for channel in channels:
		master_path_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_path_dict[channel][file.split("_")[0]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file
				
	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
	}

	channel_list_dict = {}
	for channel in channels:
		channel_list_dict[channel+'_pixel_intensity'] = []

	# test
	test_path = os.path.join(inpath, channels[0])
	test_list = os.listdir(test_path)
	test_mask = io.imread(os.path.join(test_path, test_list[0]))

	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
					else:
						continue
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped', end='', flush=True)
					pass

				## for each channel
				for channel in channels:

					# load image and mask (for given channel and imageno)
					img_path = os.path.join(inpath, channel, master_path_dict[channel][imageno])
					image = io.imread(img_path)

					pixel_intensity = list(image[roi_mask])

					# append channel lists
					channel_list_dict[channel+'_pixel_intensity'].append(pixel_intensity)

				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(channel_list_dict)
	df = pd.concat([series1, series2], axis=1)

	df.to_csv(os.path.join(inpath, 'line_ROI_pixel_intensity.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)
	
def analyze_stains(inpath, channels, ROIs):
	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	# make dictionaries

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	for channel in channels:
		master_path_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_path_dict[channel][file.split("_")[0]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file
				
	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
	}

	channel_list_dict = {}
	for channel in channels:
		channel_list_dict[channel+'_pixel_count'] = []
		channel_list_dict[channel+'_total_intensity'] = []
		channel_list_dict[channel+'_connected_component_count'] = [] # 24_08_20						

	# test
	test_path = os.path.join(inpath, channels[0], 'labels')
	test_file = os.listdir(test_path)[0]
	test_mask = io.imread(os.path.join(test_path, test_file))

	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'polygon':
						rr, cc = draw.polygon(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'rectangle':
						rr, cc = draw.rectangle(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped', end='', flush=True)
					continue

				## for each channel
				for channel in channels:
				
					# load image and mask (for given channel and imageno)
					img_path = os.path.join(inpath, channel, master_path_dict[channel][imageno])
					mask_path = os.path.join(inpath, channel, 'labels', master_path_dict[channel][imageno])
					image = io.imread(img_path)
					mask = io.imread(mask_path)
					mask = mask==255

					# find intersection of channel mask and roi mask
					inter_mask = roi_mask & mask

					pixel_count = np.count_nonzero(inter_mask)
					total_intensity = sum(image[inter_mask])
					
					# 24_08_31	
					labeled_image = measure.label(inter_mask, connectivity=2)
					props = measure.regionprops(labeled_image)
					CCcount = 0
					for prop in props:
						if prop.area >= 3: # only count connected components with size greater than 5 pixels
							CCcount += 1				 

					# append channel lists
					channel_list_dict[channel+'_pixel_count'].append(pixel_count)
					channel_list_dict[channel+'_total_intensity'].append(total_intensity)
					channel_list_dict[channel+'_connected_component_count'].append(CCcount) # 24_08_31				  

				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(channel_list_dict)
	df = pd.concat([series1, series2], axis=1)

	df.to_csv(os.path.join(inpath, 'stain_analysis.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)

def analyze_cellSize(inpath, channels, ROIs, combine_masks):
	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	# make dictionaries

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	for channel in channels:
		master_path_dict[channel] = {} # subdictionary containing image#: filename
		for file in os.listdir(os.path.join(inpath, channel)):
			master_path_dict[channel][file.split("_")[0]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file
				
	if len(combine_masks)==2:
		combined_mask_channel = combine_masks[0]+'_'+combine_masks[1]+'_labels'
		master_path_dict[combined_mask_channel] = {}
		for file in os.listdir(os.path.join(inpath, combined_mask_channel)):
			master_path_dict[combined_mask_channel][file.split("_")[0]] = file
				
	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
	}

	channel_list_dict = {}
	for channel in channels:
		channel_list_dict[channel+'_pixel_count'] = []
		channel_list_dict[channel+'_total_intensity'] = []
		channel_list_dict[channel+'CC1_pixel_count'] = []
		channel_list_dict[channel+'CC1_total_intensity'] = []
		channel_list_dict[channel+'CC1_convex_hull'] = []
	if len(combine_masks)==2:
		channel_list_dict[combined_mask_channel+'_pixel_count'] = []
		channel_list_dict[combined_mask_channel+'CC1_pixel_count'] = []
		channel_list_dict[combined_mask_channel+'CC1_convex_hull'] = []
	# test
	test_path = os.path.join(inpath, channels[0], 'labels')
	test_file = os.listdir(test_path)[0]
	test_mask = io.imread(os.path.join(test_path, test_file))

	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'polygon':
						rr, cc = draw.polygon(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'rectangle':
						rr, cc = draw.rectangle(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped', end='', flush=True)
					continue

				## for each channel
				for channel in channels:
					# load image and mask (for given channel and imageno)
					img_path = os.path.join(inpath, channel, master_path_dict[channel][imageno])
					mask_path = os.path.join(inpath, channel, 'labels', master_path_dict[channel][imageno])
					image = io.imread(img_path)
					mask = io.imread(mask_path)
					mask = mask==255

					# find intersection of channel mask and roi mask
					inter_mask = roi_mask & mask

					pixel_count = np.count_nonzero(inter_mask)
					total_intensity = sum(image[inter_mask])

					# append channel lists
					channel_list_dict[channel+'_pixel_count'].append(pixel_count)
					channel_list_dict[channel+'_total_intensity'].append(total_intensity)

					# find the largest connected component (CC1)
					try:
						labeled_mask, num_components = measure.label(inter_mask, connectivity=2, return_num=True)
						props = measure.regionprops(labeled_mask)
						areas = [prop.area for prop in props]
						# Finding the indices that would sort the areas in descending order
						sorted_indices = np.argsort(-np.array(areas))
						# Create a new binary mask with only the largest connected component
						largest_component_mask = np.zeros_like(inter_mask, dtype=bool)
						largest_component_mask[labeled_mask == sorted_indices[0] + 1] = True # add 1 because skimage.measure.label starts at 1, but np.array starts at 0
						# find the CC1 area and intensity
						channel_list_dict[channel+'CC1_pixel_count'].append(np.count_nonzero(largest_component_mask))
						channel_list_dict[channel+'CC1_total_intensity'].append(sum(image[largest_component_mask]))
					except IndexError:
						channel_list_dict[channel+'CC1_pixel_count'].append('error')
						channel_list_dict[channel+'CC1_total_intensity'].append('error')
						print('index error detected: check segmentation', end='', flush=True)
					# find the CC1 convex hull area
					try:
						hulls = [prop.area_convex for prop in props]
						channel_list_dict[channel+'CC1_convex_hull'].append(max(hulls))		
					except:
						channel_list_dict[channel+'CC1_convex_hull'].append('error')
						#print('hull error', end='', flush=True)						   
		
				# work on combined mask			  
				if len(combine_masks)==2:
					mask_path = os.path.join(inpath, combined_mask_channel, master_path_dict[combined_mask_channel][imageno])
					mask = io.imread(mask_path)
					mask = mask==255
					inter_mask = roi_mask & mask
					
					pixel_count = np.count_nonzero(inter_mask)

					# append channel lists
					channel_list_dict[combined_mask_channel+'_pixel_count'].append(pixel_count)

					try:
						labeled_mask, num_components = measure.label(inter_mask, connectivity=2, return_num=True)
						props = measure.regionprops(labeled_mask)
						areas = [prop.area for prop in props]
						# Finding the indices that would sort the areas in descending order
						sorted_indices = np.argsort(-np.array(areas))
						# Create a new binary mask with only the largest connected component
						largest_component_mask = np.zeros_like(inter_mask, dtype=bool)
						largest_component_mask[labeled_mask == sorted_indices[0] + 1] = True # add 1 because skimage.measure.label starts at 1, but np.array starts at 0
						# find the CC1 area and intensity
						channel_list_dict[combined_mask_channel+'CC1_pixel_count'].append(np.count_nonzero(largest_component_mask))
					except IndexError:
						channel_list_dict[combined_mask_channel+'CC1_pixel_count'].append('error')
						print('index error detected: check input ROIs and masks', end='', flush=True)
					# find the CC1 convex hull area
					try:
						hulls = [prop.area_convex for prop in props]
						channel_list_dict[combined_mask_channel+'CC1_convex_hull'].append(max(hulls))		
					except:
						channel_list_dict[combined_mask_channel+'CC1_convex_hull'].append('error')
						#print('error detected: check input ROIs and masks', end='', flush=True)		

				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(channel_list_dict)
	df = pd.concat([series1, series2], axis=1)

	df.to_csv(os.path.join(inpath, 'cellSize_analysis.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)
	
def metaphase_spread_segmentation(inpath):			
	label_dir = os.path.join(inpath, 'labels')
	if not os.path.isdir(label_dir):
		os.mkdir(label_dir)

	img_list = os.listdir(inpath)
	for file in img_list:
		if file.endswith('.png') or file.endswith('.tif') or file.endswith('.jpg'):
			color_image = cv2.imread(os.path.join(inpath, file))
			b,g,r = cv2.split(color_image)
			
			segmented = []
			try:# assuming b is dapi
				thresholds = threshold_multiotsu(b)
				segments = np.digitize(b, bins=thresholds)
				segmented_dapi = segments>=np.max(segments)-1 #	 low thresh
			except ValueError: # if there are only 2 values in the image, cannot threshold into 3 classes
				thresholds = threshold_otsu(b)
				segmented_dapi = b > thresholds
			segmented_dapi = segmented_dapi.astype('uint8')*255
			segmented.append(segmented_dapi)

			for fish in [g,r]: # assuming one of these is FISH
				if np.sum(fish)>0:
					try:
						thresholds = threshold_multiotsu(fish)
						segments = np.digitize(fish, bins=thresholds)
						segmented_fish = segments>=np.max(segments)-0 #	 high thresh (for lower thresh can try 1)
					except ValueError: # if there are only 2 values in the image, cannot threshold into 3 classes
						thresholds = threshold_otsu(fish)
						segmented_fish = fish > thresholds
					segmented_fish = segmented_fish.astype('uint8')*255
					segmented.append(segmented_fish)
			
			# combine the dapi and fish masks
			inter_mask = segmented[0] | segmented[1]
			inter_mask = morphology.area_closing(inter_mask, area_threshold=2977, connectivity=1, parent=None, tree_traverser=None)
			
			imagenumber = file.split('_')[0]
			cv2.imwrite(os.path.join(label_dir, imagenumber+'_combined_mask.png'), inter_mask)
				   
			print('.', end='', flush=True)

def metaphase_spread_analysis(inpath, ROIs):
	label_dir = os.path.join(inpath, 'labels')
	roi_path = os.path.join(inpath, 'ROIs')
	roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
		# if type == polygon, coordinates = (array of rows, array of cols)
		# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
		# if type == line, coordinates = (row0, col0, ro1, col1)

	master_path_dict = {} # dictionary of channels, each of which is a dictionary
	master_path_dict['labels'] = {} # subdictionary containing image#: filename
	for file in os.listdir(label_dir):
		master_path_dict['labels'][file.split("_")[0]] = file
	for roi_name in ROIs:
		master_path_dict['roi_'+roi_name] = {}
		for file in os.listdir(roi_path):
			if roi_name in file:
				master_path_dict['roi_'+roi_name][file.split("_")[0]] = file

	analysis_dict = {
		'Image_no': [],
		'ROI_name': [],
		'ROI_no': [],
	}
	
	stats_dict = {}
	stats_dict['pixel_count'] = []
	stats_dict['connected_component_count'] = []
	stats_dict['convex_hull_area'] = []

	# test
	test_file = os.listdir(label_dir)[0]
	test_mask = io.imread(os.path.join(label_dir, test_file))
	
	## for each ROI name
	for roi_name in ROIs:
			
		## for each ROI file
		for imageno in master_path_dict['roi_'+roi_name].keys():

			# unpickle ROI file
			with open(os.path.join(roi_path, master_path_dict['roi_'+roi_name][imageno]), "rb") as fp:	 # Unpickling
				roi_coordinates[roi_name] = pickle.load(fp)

			## for each ROI inside roi file with roi name	 
			for roi_index, roi_object in enumerate(roi_coordinates[roi_name]):
				
				# make mask from ROI:
				roi_mask = np.zeros(test_mask.shape, dtype=bool)
				roi, coordi = roi_object
				try:
					if roi == 'polygon':
						rr, cc = draw.polygon(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'rectangle':
						rr, cc = draw.rectangle(coordi[0], coordi[1])
						roi_mask[rr, cc] = True
					elif roi == 'line':
						rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
						roi_mask[rr, cc] = True
				except:
					print('ROI '+str(roi_index)+' in '+roi_name+' '+str(imageno)+' was skipped', end='', flush=True)
					continue
			
				# load image and mask (for given channel and imageno)
				mask_path = os.path.join(label_dir, master_path_dict['labels'][imageno])
				mask = io.imread(mask_path)
				mask = mask==255

				# find intersection of channel mask and roi mask
				inter_mask = roi_mask & mask

				pixel_count = np.count_nonzero(inter_mask)
				#labeled_mask, num_components = measure.label(inter_mask, connectivity=2, return_num=True)
				labeled_image = measure.label(inter_mask, connectivity=2)
				props = measure.regionprops(labeled_image)
				CCcount = 0
				for prop in props:
					if prop.area >= 5: # only count connected components with size greater than 5 pixels
						CCcount += 1
				convex_hull = morphology.convex_hull_image(inter_mask)
				convex_hull_area = np.count_nonzero(convex_hull)

				# append channel lists
				stats_dict['pixel_count'].append(pixel_count)
				stats_dict['connected_component_count'].append(CCcount)
				stats_dict['convex_hull_area'].append(convex_hull_area)				   

				# append lists
				analysis_dict['Image_no'].append(imageno)
				analysis_dict['ROI_name'].append(roi_name)
				analysis_dict['ROI_no'].append(roi_index) ############## index
			print('.', end='', flush=True)

	series1 = pd.DataFrame(analysis_dict)
	series2 = pd.DataFrame(stats_dict)
	df = pd.concat([series1, series2], axis=1)

	df.to_csv(os.path.join(inpath, 'metaphase_spread_analysis.csv'), index=False)
	print('CSV file saved to '+inpath, end='', flush=True)