#!/usr/bin/env python3

#loops through all the subdirectories in given directory and runs 
import yaml
import os
from utils import *

config = open("ROI_selector_config.yaml")
var = yaml.load(config, Loader=yaml.FullLoader)['path']
superpath = var['directory'] # when 'inpath' directed is a dir with subdirs
channels = var['channels']
combine_channels = var['combine_channels']
combine_colors = var['combine_colors']
to_run = var['to_run']
high_thresh = var['high_thresh'] # highest segment of multi-otsu only
low_thresh = var['low_thresh'] # top 2 highest segments of multi-otsu
combine_masks = var['combine_masks']
close_holes = var['close_holes']

chrDNA_channel = var['chrDNA_channel']
ecDNA_channel = var['ecDNA_channel']
no_nuclei = var['no_nuclei']
expand_chrDNA_pixel = var['expand_chrDNA_pixel']
ROIs = var['ROIs']

print('initiating...', end='', flush=True)	
for dir in os.listdir(superpath):
	if os.path.isdir(os.path.join(superpath, dir)):
		inpath = os.path.join(superpath, dir)
		if 'sort' in to_run:
			print('\nsorting ' + dir, end='', flush=True)
			sort_channel(inpath, channels)
		if 'combine' in to_run:
			print('\ncombining ' + dir, end='', flush=True)
			combine_channel(inpath, combine_channels, combine_colors)
		if 'segment' in to_run:
			print('\nsegmenting ' + dir, end='', flush=True)
			stain_segmentation(inpath, channels, high_thresh, low_thresh, combine_masks, close_holes)
		if 'analyze_untethering' in to_run:
			print('\nanalyzing untethering ' + dir, end='', flush=True)
			analyze_untethering(inpath, channels, chrDNA_channel, ecDNA_channel, no_nuclei, expand_chrDNA_pixel, ROIs, combine_masks)
		if 'analyze_micronuclei' in to_run:
			print('\nanalyzing micronuclei ' + dir, end='', flush=True)
			analyze_micronuclei(inpath, channels, chrDNA_channel, ecDNA_channel, no_nuclei, expand_chrDNA_pixel, ROIs, combine_masks)
		if 'analyze_cellSize' in to_run:
			print('\nanalyzing cell size ' + dir, end='', flush=True)
			analyze_cellSize(inpath, channels, ROIs, combine_masks)
		if 'segment_metaphase_spread' in to_run:
			print('\nsegmenting spreads ' + dir, end='', flush=True)
			metaphase_spread_segmentation(inpath)
		if 'analyze_metaphase_spread' in to_run:
			print('\nanalyzing spreads ' + dir, end='', flush=True)
			metaphase_spread_analysis(inpath, ROIs)
		if 'analyze_stains' in to_run:
			print('\nanalyzing stains ' + dir, end='', flush=True)
			analyze_stains(inpath, channels, ROIs)
		if 'analyze_line' in to_run:
			print('\nanalyzing lines ' + dir, end='', flush=True)
			line_intensity_analysis(inpath, channels, ROIs)
	else:
		print(dir+" is not a directory...skipping...")
print('done')