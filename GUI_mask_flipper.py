#!/usr/bin/env python
# coding: utf-8

# # ROI selector GUI

# In[1]:


version_no = "4"


# ### change log]
# v4: added ability to separate color image by channel
# v3: incorporated ability to adjust masks a la ecSeg and mnSeg GUIs
# v2: load dapi based on image# (first part of filename, rather than the entire filename)
# v1: based on mnSeg GUI v3

# In[2]:


# python 3.8.5, skimage 0.18.1
# env = cv2

from tkinter import *
from tkinter import ttk, filedialog
from tkinter import messagebox

from PIL import ImageGrab, Image, ImageTk

import yaml
import os
import pickle # for saving roi boxes boundaries

import difflib # for get close matches
				
import numpy as np
from skimage import measure, draw, io, img_as_ubyte
from skimage.color import gray2rgb #v3

#v3:
import pathlib # for making temp directory
import glob # for temp mask files
import datetime # for naming temp mask files

# In[3]:


###### intialize ######
config = open("ROI_selector_config.yaml")
var = yaml.load(config, Loader=yaml.FullLoader)['path']
inpath = var['directory']

LABELS = ['blue','yellow', 'green', 'red', 'white', 'black'] #v3
# blue = [56, 108, 176]
# yellow = [255, 255, 153]
# green = [127, 201, 127]
# red = [240, 2, 127]
# black = [0,0,0]
			
all_files = []

# for ROI selection
ycoor = []
xcoor = []

temp_mask_values = [200, 200, 200] # default color of anything drawn on temp mask

# for polygon mask flip tool #v3
flipycoor = []
flipxcoor = []

def filename_counter(counter): #v3
	if counter < 10:
		fname = "000"+str(counter)
	elif counter < 100:
		fname = "00"+str(counter)
	elif counter < 1000:
		fname = "0"+str(counter)
	else:
		fname = str(counter)
	return fname

switches = {} #v3
# flip_from
# flip_to
# display_masks

image_dict = {}
# image0 is the original, never manipulate once open
# roi_overlay is manipulated with overlaid masks instead
# image1 is the display image

mask_dict = {}
# mask_image is the original, never manipulate once created or loaded. Resets only when new image is opened

thresh_tracker = {} # v3
# format: channel:threshold
# this gets reset for each new image

thresh_dict = {} # v3
# format: image_name:[r, g, b]
# this is used for all images

# ROIs
ROIs = var['ROIs'] # names of ROIs (defined in config.yaml file)
roi_coordinates = {} # each ROI name gets a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)
	# if type == polygon, coordinates = (array of rows, array of cols)
	# if type == rectangle, coordinates = ((row0,col0), (row1,col1))
	# if type == line, coordinates = (row0, col0, ro1, col1)
roi_colorkeys = {} 
roi_colorvalues = {} # each ROI name gets a color value; this is set during initiation

color_palette = {
	'red':[255, 0, 0],
	'green':[0, 255, 0],
	'cyan':[0,255,255],
	'magenta':[255,0,255],
	'brown':[165,42,42],
	'purple':[128,0,128],
	'navy':[0, 0, 128],
	'orange':[255,165,0],
	'beige':[245,245,220],
	'yellow':[255,255,0],
}

for i, name in enumerate(ROIs):
	roi_coordinates[name] = []
	roi_colorkeys[name] = list(color_palette.keys())[i%len(color_palette)]
	roi_colorvalues[name] = color_palette[roi_colorkeys[name]]

# load file lists
subdirs = {'dapi': os.path.join(inpath, 'dapi'),
		   'mask': os.path.join(inpath, 'labels'),
		   'roi': os.path.join(inpath, 'ROIs')}
dapi_dict = {} # v2
#mask_dict = {} # v4
roi_dict = {} # v4
try:
	for file in os.listdir(subdirs['dapi']): # v2
		dapi_dict[file.split('_')[0]] = file
except:
	pass
try:
	for file in os.listdir(subdirs['mask']): # v4
		if file.startswith('updated_'):
			mask_dict[file.split('_')[1]] = file
		else:
			mask_dict[file.split('_')[0]] = file
except:
	pass
try:
	for name in ROIs:
		roi_dict[name] = {}
		for file in os.listdir(subdirs['roi']): # v4
			if name in file:
				roi_dict[name][file.split('_')[0]] = file
except:
	pass
###### initialize ######

###########################v3###################################
def update_mask(save_temp=False):
	if 'display_masks' not in switches:
		image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['color_image']))
		canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
		
	else:
		image_dict['mask_overlay'] = image_dict['color_image'].copy()
		
		for i in switches['display_masks']:
			if i == 0:
				image_dict['mask_overlay'][mask_dict['blue_mask']] = [56, 108, 176]
			elif i == 1:
				image_dict['mask_overlay'][mask_dict['yellow_mask']] = [255, 255, 153]
			elif i == 2:
				image_dict['mask_overlay'][mask_dict['green_mask']] = [127, 201, 127]
			elif i == 3:
				image_dict['mask_overlay'][mask_dict['red_mask']] = [240, 2, 127]
			elif i == 5:
				image_dict['mask_overlay'][mask_dict['black_mask']] = [0, 0, 0]
			elif i == 4:
				image_dict['mask_overlay'][mask_dict['white_mask']] = [255, 255, 255]					

		image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['mask_overlay']))
		canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	
	if save_temp==True: # called when masks are created, loaded, or manipulated
		pathlib.Path('temp').mkdir(parents=True, exist_ok=True)

		temp_filter = np.zeros((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), dtype=np.uint8)
		temp_filter = gray2rgb(temp_filter)
		temp_filter[mask_dict['blue_mask']] = [56, 108, 176]
		temp_filter[mask_dict['yellow_mask']] = [255, 255, 153]
		temp_filter[mask_dict['green_mask']] = [127, 201, 127]
		temp_filter[mask_dict['red_mask']] = [240, 2, 127]
		temp_filter[mask_dict['black_mask']] = [0, 0, 0]
		temp_filter[mask_dict['white_mask']] = [255, 255, 255]

		io.imsave(os.path.join('temp', 'temp_mask'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.png'),
				  temp_filter, check_contrast=False)

# funtion to be called when "Polygon select" button is clicked
def select_polygon(*args):	  
	canvas.bind("<ButtonPress-1>", savecoords)
	canvas.bind("<Double-Button-1>", mask_set) # double clicking ends coords collection and calculates

def savecoords(event):
	
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	
	# prevent out of bounds event
	if cx < 0:
		cx = 0
	if cx >= image_dict['image0'].shape[1]:
		cx = image_dict['image0'].shape[1]-1
	if cy < 0:
		cy = 0
	if cy >= image_dict['image0'].shape[0]:
		cy = image_dict['image0'].shape[0]-1
		
	flipxcoor.append(int(cx))
	flipycoor.append(int(cy))

	if len(flipxcoor)==1:
		rr, cc = draw.disk((flipycoor[0], flipxcoor[0]), 3)
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(overlay='mask_overlay')
	elif len(flipxcoor)>1:
		rr, cc = draw.line(flipycoor[-2], flipxcoor[-2], flipycoor[-1], flipxcoor[-1])
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(overlay='mask_overlay')

def mask_set(event): 
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	flipxcoor.append(cx)
	flipycoor.append(cy)
	r = np.array(flipycoor)
	c = np.array(flipxcoor)
	rr, cc = draw.polygon(r, c)
	
	# reset
	flipxcoor.clear()
	flipycoor.clear()
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	
	flip_masks(switches['flip_from'], switches['flip_to'], rr, cc)

def flip_masks(flip_from, flip_to, rr, cc):
	for i in flip_from:
		if i == 0:
			mask_dict['blue_mask'][rr,cc] = False
		elif i == 1:
			mask_dict['yellow_mask'][rr,cc] = False
		elif i == 2:
			mask_dict['green_mask'][rr,cc] = False
		elif i == 3:
			mask_dict['red_mask'][rr,cc] = False
		elif i == 5:
			mask_dict['black_mask'][rr,cc] = False
		elif i == 4:
			mask_dict['white_mask'][rr,cc] = False

	temp_filter = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), True)
	temp_filter[mask_dict['blue_mask']] = False
	temp_filter[mask_dict['yellow_mask']] = False
	temp_filter[mask_dict['green_mask']] = False
	temp_filter[mask_dict['red_mask']] = False
	temp_filter[mask_dict['black_mask']] = False
	temp_filter[mask_dict['white_mask']] = False
	
	for i in flip_to:
		if i == 0:
			mask_dict['blue_mask'][temp_filter] = True
		elif i == 1:
			mask_dict['yellow_mask'][temp_filter] = True
		elif i == 2:
			mask_dict['green_mask'][temp_filter] = True
		elif i == 3:
			mask_dict['red_mask'][temp_filter] = True
		elif i == 5:
			mask_dict['black_mask'][temp_filter] = True
		elif i == 4:
			mask_dict['white_mask'][temp_filter] = True

	update_mask(save_temp=True)
	
def dot_ec(*args):	
	canvas.bind("<ButtonPress-1>", draw_dot)

def draw_dot(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	rr, cc = draw.disk((cy, cx), dot_radius.get())
	
	mask_dict['blue_mask'][rr,cc] = False
	mask_dict['yellow_mask'][rr,cc] = False
	mask_dict['green_mask'][rr,cc] = False
	mask_dict['red_mask'][rr, cc] = True
	mask_dict['black_mask'][rr,cc] = False
	mask_dict['white_mask'][rr,cc] = False

	update_mask(save_temp=True)
	
def undo_manip(*args):
	try:
		latest_temp = get_latest_file(os.path.join('temp', 'temp_mask*'))
		os.remove(latest_temp)
		latest_temp = get_latest_file(os.path.join('temp', 'temp_mask*'))
		temp_filter = io.imread(latest_temp)
		divide_masks(temp_filter)
		update_mask()
	except:
		messagebox.showinfo(message='Nothing to undo')

def get_latest_file(directory):
	temp_mask_list = glob.glob(directory)
	latest_temp = max(temp_mask_list, key=os.path.getctime)
	return latest_temp

def divide_masks(mask):
	mask_reset()
	mask_dict['blue_mask'][np.where(np.all(mask==[56, 108, 176], axis=-1))] = True
	mask_dict['yellow_mask'][np.where(np.all(mask==[255, 255, 153], axis=-1))] = True
	mask_dict['green_mask'][np.where(np.all(mask==[127, 201, 127], axis=-1))] = True
	mask_dict['red_mask'][np.where(np.all(mask==[240, 2, 127], axis=-1))] = True
	mask_dict['black_mask'][np.where(np.all(mask==[0, 0, 0], axis=-1))] = True
	mask_dict['white_mask'][np.where(np.all(mask==[255, 255, 255], axis=-1))] = True
	
def mask_reset(*args): # does not reset ecseg mask
	image_dict['mask_overlay'] = image_dict['color_image'].copy()
	# reset masks	 
	mask_dict['blue_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['yellow_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['green_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['red_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['black_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['white_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	#mask_dict['dm_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	
def clear_temp_folder(*args):
	if pathlib.Path('temp').is_dir():
		files = glob.glob('temp\*')
		for f in files:
			try:
				os.remove(f)
			except:
				pass

def save_masks(*args): # called on by function save() (defined in tkinter)
	temp_filter = np.zeros((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), dtype=np.uint8)
	temp_filter = gray2rgb(temp_filter)
	
	temp_filter[mask_dict['blue_mask']] = [56, 108, 176]
	temp_filter[mask_dict['yellow_mask']] = [255, 255, 153]
	temp_filter[mask_dict['green_mask']] = [127, 201, 127]
	temp_filter[mask_dict['red_mask']] = [240, 2, 127]
	temp_filter[mask_dict['black_mask']] = [0, 0, 0]
	temp_filter[mask_dict['white_mask']] = [255, 255, 255]

	# blue = [56, 108, 176]
	# yellow = [255, 255, 153]
	# green = [127, 201, 127]
	# red = [240, 2, 127]

	try:
		#if mass_state.get() == 0:	v12
		#	 File = filedialog.asksaveasfilename(parent=root, initialdir=inpath,
		#										 title='Save as', filetypes=[("image", ".png")])
		#	 maskfile.set(File.split('/')[-1])
		#	 io.imsave(maskfile.get(), temp_filter, check_contrast=False)
		#elif mass_state.get() == 1:
		if not maskfile.get().startswith('updated_'):
			maskfile.set('updated_'+maskfile.get())
		mask_path = os.path.join(subdirs['mask'], maskfile.get())
		io.imsave(mask_path, temp_filter, check_contrast=False)
	except:
		print('save_masks error')
		
	messagebox.showinfo('Mask saved as:', maskfile.get()) #v3
	
	# clear temp directory
	clear_temp_folder()
###########################v3###################################



# In[4]:


def update_temp(mask_values=temp_mask_values, overlay='roi_overlay'): # this is for drawing lines for polygons (temporary)
	temp_overlay = image_dict[overlay].copy()
	temp_overlay[mask_dict['temp_mask']] = mask_values
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(temp_overlay))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")


# In[5]:


def delete_rois(*args): # delete all ROIs of the selected ROI
	roi_coordinates[select_roi.get()].clear()
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()
	
def reset_polygon(*args): # resets current ROI or mask flip tool (polygon, rectangle, or line)
	xcoor.clear()
	ycoor.clear()
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	update_temp()


# In[6]:


# function to be called when draw polygon button is clicked
def idraw_polygon(*args):	 
	canvas.bind("<ButtonPress-1>", start_polygon)
	canvas.bind("<Double-Button-1>", end_polygon) # double clicking ends coords collection and calculates

def start_polygon(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
		
	xcoor.append(int(cx))
	ycoor.append(int(cy))
	
	if len(xcoor)==1:
		rr, cc = draw.disk((ycoor[0], xcoor[0]), 3)
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(roi_colorvalues[select_roi.get()])
	elif len(xcoor)>1:
		rr, cc = draw.line(int(ycoor[-2]), int(xcoor[-2]), int(ycoor[-1]), int(xcoor[-1]))
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(roi_colorvalues[select_roi.get()])

def end_polygon(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	xcoor.append(cx)
	ycoor.append(cy)
	r = np.array(ycoor)
	c = np.array(xcoor)
	#rr, cc = draw.polygon(r, c)
	
	roi_coordinates[select_roi.get()].append(('polygon', (r, c)))
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()
	
	# reset temp
	xcoor.clear()
	ycoor.clear()
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)

def undraw_polygon(*args):
	canvas.bind("<ButtonPress-1>", remove_polygon)

def remove_polygon(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	
	point = [cy, cx] # r, c
	for (roi,coordi) in roi_coordinates[select_roi.get()]: # need to improve efficiency
		if roi=='polygon':
			vertices = np.column_stack(coordi)
			if measure.points_in_poly([point], vertices)[0]:
				roi_coordinates[select_roi.get()].remove((roi,coordi))
				break
		  
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()


# In[7]:


# function to be called when draw rectangle button is clicked
def idraw_rectangle(*args):	   
	canvas.bind("<ButtonPress-1>", start_rectangle)

def start_rectangle(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
		
	xcoor.append(int(cx))
	ycoor.append(int(cy))
	
	if len(xcoor)==1:
		rr, cc = draw.disk((ycoor[0], xcoor[0]), 3)
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(roi_colorvalues[select_roi.get()])
	elif len(xcoor)==2:
		end_rectangle()
		
def end_rectangle(*args):
	#rr, cc = draw.rectangle((ycoor[0], xcoor[0]), (ycoor[1], xcoor[1]))
	start = (ycoor[0], xcoor[0])
	end = (ycoor[1], xcoor[1])
	roi_coordinates[select_roi.get()].append(('rectangle', (start, end)))
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()
	
	# reset temp
	xcoor.clear()
	ycoor.clear()
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)

def undraw_rectangle(*args):
	canvas.bind("<ButtonPress-1>", remove_rectangle)

def remove_rectangle(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	
	point = [cy, cx] # r, c
	for (roi,coordi) in roi_coordinates[select_roi.get()]:
		if roi=='rectangle':
			start = coordi[0]
			end = coordi[1]
			vertices = np.array([
						(max(start[0], end[0]), min(start[1], end[1])), # ul
						(max(start[0], end[0]), max(start[1], end[1])), # ur
						(min(start[0], end[0]), max(start[1], end[1])), # lr
						(min(start[0], end[0]), min(start[1], end[1])), # ll
						]) 

			if measure.points_in_poly([point], vertices)[0]:
				roi_coordinates[select_roi.get()].remove((roi,coordi))
				break
	
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()


# In[8]:


# function to be called when draw line button is clicked
def idraw_line(*args):	  
	canvas.bind("<ButtonPress-1>", start_line)

def start_line(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
		
	xcoor.append(int(cx))
	ycoor.append(int(cy))
	
	if len(xcoor)==1:
		rr, cc = draw.disk((ycoor[0], xcoor[0]), 3)
		mask_dict['temp_mask'][rr, cc] = True
		update_temp(roi_colorvalues[select_roi.get()])
	elif len(xcoor)==2:
		end_line()
		
def end_line(*args):
	#rr, cc = draw.line(ycoor[0], xcoor[0], ycoor[1], xcoor[1])
	
	roi_coordinates[select_roi.get()].append(('line', (ycoor[0], xcoor[0], ycoor[1], xcoor[1])))
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()
	
	# reset temp
	xcoor.clear()
	ycoor.clear()
	mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)

def undraw_line(*args):
	canvas.bind("<ButtonPress-1>", remove_line)

def remove_line(event):
	event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y)) # converts event (window) coordinates to image coordinates
	cx, cy = event2canvas(event, canvas)
	
	point = [cy, cx] # r, c
	for (roi,coordi) in roi_coordinates[select_roi.get()]: # need to improve efficiency
		if roi=='line':
			# pick the left, top most point
			if coordi[1]<coordi[3]:
				startpoint = (coordi[0], coordi[1])
			elif coordi[1]>coordi[3]:
				startpoint = (coordi[2], coordi[3])
			elif coordi[0]>coordi[2]:
				startpoint = (coordi[0], coordi[1])
			else:
				startpoint = (coordi[2], coordi[3])
				
			radius = 5
			segment = (radius/2)**.5
			vertices = np.array([
						(startpoint[0]+radius, startpoint[1]), # top
						(startpoint[0]+segment, startpoint[1]+segment), # top right
						(startpoint[0], startpoint[1]+radius), # right
						(startpoint[0]-segment, startpoint[1]+segment), # bottom right
						(startpoint[0]-radius, startpoint[1]), # bottom
						(startpoint[0]-segment, startpoint[1]-segment), # bottom left
						(startpoint[0], startpoint[1]-radius), # left
						(startpoint[0]+segment, startpoint[1]-segment), # top left
						]) 
			if measure.points_in_poly([point], vertices)[0]:
				roi_coordinates[select_roi.get()].remove((roi,coordi))
				break
		  
	#roi_count.set(len(roi_coordinates[select_roi.get()]))
	update_roi()


def update_roi():
	image_dict['roi_overlay'] = image_dict['color_image'].copy()

	for name in ROIs:
		for (roi,coordi) in roi_coordinates[name]:
			if roi == 'polygon':
				try:
					rr, cc = draw.polygon_perimeter(coordi[0], coordi[1])
					image_dict['roi_overlay'][rr, cc] = roi_colorvalues[name]
				except IndexError:
					reset_polygon
					roi_coordinates[select_roi.get()] = roi_coordinates[select_roi.get()][:-1] # remove the last poorly drawn polygon
			elif roi == 'rectangle':
				rr, cc = draw.rectangle_perimeter(coordi[0], coordi[1])
				image_dict['roi_overlay'][rr, cc] = roi_colorvalues[name]
			elif roi == 'line':
				rr, cc = draw.line(coordi[0], coordi[1], coordi[2], coordi[3])
				image_dict['roi_overlay'][rr, cc] = roi_colorvalues[name]

	show_roi()
	
def hard_reset(reset_image=False, reset_roi=False):
	if reset_image:
		# reset image
		image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['image0']))
		canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")

	if reset_roi:
		image_dict['roi_overlay'] = image_dict['color_image'].copy()
		for name in ROIs:
			roi_coordinates[name].clear()
		#roi_count.set(len(roi_coordinates[select_roi.get()]))
		
		# create temp mask for showing current working ROI
		mask_dict['temp_mask'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False)
	
def show_original(*args):  
	temp_overlay = image_dict['color_image'].copy()
			
	temp_overlay[mask_dict['temp_mask']] = temp_mask_values
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(temp_overlay))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")

def show_roi(*args):
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['roi_overlay']))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	
def show_dapi_image(*args):
	try:
		image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['dapi_image']))
		canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	except:
		pass
		
def show_mask_image(*args):
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['mask_overlay']))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")

def populate_image_files(*args):
	all_files.clear()
	populated.set(False) # v13
	for file in os.listdir(inpath):
		if file.endswith(".PNG") or file.endswith(".png") or file.endswith(".TIF") or file.endswith(".tif") or file.endswith(".JPG") or file.endswith(".jpg"):
			if continue_state.get() == 1:
				if file[:-4] not in [x.split('_')[0] for x in os.listdir(subdirs['roi'])]:
					all_files.append(file)
			elif continue_state.get() == 0:
				all_files.append(file)
	
def open_file(*args):
	if populated.get(): # v13 #v17
		populate_image_files()
	if mass_state.get() == 0:
		File = filedialog.askopenfilename(parent=root, initialdir=image_path.get(), # v13
								title='Select image file to open')#, filetypes=[("image", ".png", ".tif")])
		#print("opening %s" % File)
		imgfile.set(os.path.split(File)[-1]) #v17
		file_number.set(all_files.index(imgfile.get())+1)
	elif mass_state.get() == 1:
		File = os.path.join(image_path.get(), all_files[file_number.get()])
		imgfile.set(all_files[file_number.get()])
		file_number.set(file_number.get()+1)
		if file_number.get() == len(all_files):
			messagebox.showinfo(message='This will be the last image') #v13
	#if mass_state.get() == 0:
	#	File = filedialog.askopenfilename(parent=root, initialdir=inpath, # v13
	#							title='Select image file to open')#, filetypes=[("image", ".png", ".tif")])
	#elif mass_state.get() == 1:
	#	if populated.get():
	#		populate_image_files()
	#	File = os.path.join(inpath, all_files[file_number.get()])
	#	file_number.set(file_number.get()+1)
	#	if file_number.get() == len(all_files):
	#		messagebox.showinfo(message='This will be the last image')

	imgfile.set(os.path.split(File)[-1])
	image_dict['image0'] = io.imread(File) # must be .png or .tif
	if image_dict['image0'].dtype != 'uint8':
		image_dict['image0'] = img_as_ubyte(image_dict['image0'])
	image_dict['threshold'] = np.full((image_dict['image0'].shape[0], image_dict['image0'].shape[1]), False) # v3

	if os.path.isdir(subdirs['dapi']):
		dapi_image = dapi_dict[imgfile.get().split('_')[0]]
		image_dict['dapi_image'] = io.imread(os.path.join(subdirs['dapi'], dapi_image))
		if image_dict['dapi_image'].dtype != 'uint8':
			image_dict['dapi_image'] = img_as_ubyte(image_dict['dapi_image'])

	if len(np.shape(image_dict['image0'])) == 3:
		image_dict['image0'] = image_dict['image0'][:,:,:3] # make sure the image only has 3 color channels (no alpha)
	
	divide_channels(update=False)
	hard_reset(reset_image=True, reset_roi=True)
	clear_temp_folder() #v3
	mask_reset() #v3
	mask_dict.pop('mask_image', None)
	
	canvas.config(scrollregion=canvas.bbox(ALL), width=image_dict['image1'].width(), height=image_dict['image1'].height()) # initiates window with adjusted size to image_dict['image1']
	load_masks_and_rois()	

def load_masks_and_rois(*args):
	if 'image0' not in image_dict:
		messagebox.showinfo(message='Open an image file first')
		return
	
	try:
		if os.path.isdir(subdirs['mask']):
			updated_mask_path = os.path.join(subdirs['mask'], 'updated_'+mask_dict[imgfile.get().split('_')[0]])
			mask_path = os.path.join(subdirs['mask'], mask_dict[imgfile.get().split('_')[0]])
			if os.path.exists(updated_mask_path):
				maskfile.set(os.path.split(updated_mask_path)[-1])
				image_dict['mask_image'] = io.imread(updated_mask_path)
			elif os.path.exists(mask_path):
				maskfile.set(os.path.split(mask_path)[-1]) 
				image_dict['mask_image'] = io.imread(mask_path)
			if len(image_dict['mask_image'].shape)==2: # if mask is single channel
				image_dict['mask_image'] = gray2rgb(image_dict['mask_image'])
			elif image_dict['mask_image'].shape[2]>=3:
				image_dict['mask_image'] = image_dict['mask_image'][:,:,:3]
			else:
				print('mask only has 2 channels!')
			
			if image_dict['mask_image'].dtype != 'uint8':
				image_dict['mask_image'] = img_as_ubyte(image_dict['mask_image'])
		
		divide_masks(image_dict['mask_image'])
	except KeyError:
		maskfile.set('no mask found')
	
	if os.path.isdir(subdirs['roi']) and os.listdir(subdirs['roi']): # if ROI dir exists and is not tempty
		try:
			for name in ROIs:
				roi_file = roi_dict[name][imgfile.get().split('_')[0]]
				roifile.set(roi_file.split('_')[0])
				with open(os.path.join(subdirs['roi'], roi_file), "rb") as fp:	# Unpickling
					roi_coordinates[name] = pickle.load(fp)
		except KeyError:
			roifile.set('no ROI found')
	update_mask(save_temp=True)
	update_roi()

def save_rois(*args): # not used here
	if os.path.isdir(subdirs['roi']):
		pass
	else:
		os.mkdir(subdirs['roi'])
	for name in ROIs:
		if roi_coordinates[name]: # if not empty
			roi_file = roi_dict[name][imgfile.get().split('_')[0]]
			with open(os.path.join(subdirs['roi'], roi_file), "wb") as fp:	#Pickling
				pickle.dump(roi_coordinates[name], fp)
	
	messagebox.showinfo('ROIs saved as:', roi_file) #v3
	
	threshold_values = []
	for i in range(3):
		try:
			threshold_values.append(thresh_tracker[i]) # r, g, b
		except:
			threshold_values.append(np.nan)
		thresh_dict[imgfile.get()] = threshold_values
	#thresh_dict.to_csv(os.path.join(inpath, 'thresholds.csv'), header=['red', 'green', 'blue'])
	with open(os.path.join(inpath, 'threshold_dictionary'), "wb") as fp:   #Pickling
		pickle.dump(thresh_dict, fp)

def divide_channels(update=True):
	image_dict['color_image'] = np.zeros(image_dict['image0'].shape, dtype=np.uint8)
	for i in channel_select.curselection():
		image_dict['color_image'][:,:,i] = image_dict['image0'][:,:,i]
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['color_image'])) #v4
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	
	if update:
		update_roi() # since update_roi uses the color_image
		update_mask() # since update_roi uses the color_image

# Tkinter GUI
root = Tk()
root.title('ROI selector GUI')
#root.geometry('280x400-1+3')

mainframe = ttk.Panedwindow(root, orient=HORIZONTAL) # canvas is preventing horizontal placement of canvas child
mainframe.pack(fill=BOTH, expand=1)
mainframe.grid_columnconfigure(0, weight=1) # expand frame to fill extra space if window is resized
mainframe.grid_rowconfigure(0, weight=1)

# image frame
image_frame = ttk.Frame(mainframe) # doesn't matter what this widget is (Label, Frame, etc) because canvas takes over
image_frame.pack(fill=BOTH, expand=1)
image_frame.grid_columnconfigure(0, weight=1) # expand frame to fill extra space if window is resized
image_frame.grid_rowconfigure(0, weight=1)
mainframe.add(image_frame)

xscroll = Scrollbar(image_frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(image_frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(image_frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
image_frame.pack(fill=BOTH,expand=1)

# paned windows for widgets (inside mainframe)
paned = ttk.Panedwindow(mainframe, orient=HORIZONTAL)
paned.pack(fill=BOTH, expand=1)
mainframe.add(paned)

subpaned1 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned1)
acq_pane = ttk.Labelframe(subpaned1, text='Image acquisition')
subpaned1.add(acq_pane)
curr_pane = ttk.Labelframe(subpaned1, text='Current image')
subpaned1.add(curr_pane)

subpaned2 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned2)
channel_pane = ttk.Labelframe(subpaned2, text='Show channels')
subpaned2.add(channel_pane)

subpaned3 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned3)
masks_pane = ttk.Labelframe(subpaned3, text='Show masks')
subpaned3.add(masks_pane)

######v3######
subpaned4 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned4)
toolbar_pane = ttk.Labelframe(subpaned4, text='Flip masks')
subpaned4.add(toolbar_pane)

subpaned7 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned7)
dot_pane = ttk.Labelframe(subpaned7, text='Dot red') # v11
subpaned7.add(dot_pane) # v2
######v3######
thresh_pane = ttk.Labelframe(subpaned7, text='Set threshold')
subpaned7.add(thresh_pane)

subpaned5 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned5)
roi_pane = ttk.Labelframe(subpaned5, text='Identify ROI')
subpaned5.add(roi_pane) # v2
subpaned6 = ttk.Panedwindow(paned, orient=VERTICAL)
paned.add(subpaned6)
about_pane = ttk.Labelframe(subpaned6, text='Info')
subpaned6.add(about_pane)

#################### IMAGE ACQUISITION ######################
# Open image
ttk.Button(acq_pane, text='Open image', command=open_file).grid(column=1, row=1, sticky=(W,E))

# Save ROIs button
maskfile = StringVar()

# Mass analysis checkbox
file_number = IntVar()
file_number.set(0)
populated = BooleanVar() # v13
populated.set(True) # v13
mass_state = IntVar(value=1)
ttk.Checkbutton(acq_pane, text='Mass analysis', variable=mass_state).grid(column=1, row=3, sticky=(W,E))
# if checked, presumes there is an images subdir and masks subdir. Open will open images in order from images/, save will save masks to masks/

# Continue from where left off checkbox
continue_state = IntVar()
ttk.Checkbutton(acq_pane, text='Continue', command=populate_image_files, 
				variable=continue_state).grid(column=2, row=3, sticky=(W,E))

# Image directory
image_path = StringVar()
ttk.Label(acq_pane, text='Directory:').grid(column=1, row=4, sticky=(W,E))
id_entry = ttk.Entry(acq_pane, textvariable=image_path, width=15)
id_entry.grid(column=2, row=4, sticky=W)
id_entry.insert(0, inpath)

imgfile = StringVar()
ttk.Label(curr_pane, text='Image:').grid(column=1, row=6, sticky=(W,E))
ci_entry = ttk.Entry(curr_pane, textvariable=imgfile, width=24)
ci_entry.grid(column=2, row=6, sticky=W)
ci_entry.config(state= "disabled")

maskfile = StringVar()
ttk.Label(curr_pane, text='Mask:').grid(column=1, row=7, sticky=(W,E))
ci_entry = ttk.Entry(curr_pane, textvariable=maskfile, width=24)
ci_entry.grid(column=2, row=7, sticky=W)
ci_entry.config(state= "disabled")

roifile = StringVar()
ttk.Label(curr_pane, text='ROIs:').grid(column=1, row=8, sticky=(W,E))
ci_entry = ttk.Entry(curr_pane, textvariable=roifile, width=24)
ci_entry.grid(column=2, row=8, sticky=W)
ci_entry.config(state= "disabled")

#################### IMAGE ACQUISITION ######################

#################### SET THRESHOLD ####################
channel_list = ['red', 'green', 'blue']
channel_selection = StringVar(value=channel_list)
select_channel = Listbox(thresh_pane, listvariable=channel_selection, activestyle='none',
					 selectmode='extended', exportselection=0, width=10, height=len(channel_list))
select_channel.grid(column=1, row=1, sticky=(W,E))
#select_channel.bind("<<ListboxSelect>>", set_threshold)

channel_thresh = IntVar()
channel_thresh.set(100)
ttk.Label(thresh_pane, text='Threshold:').grid(column=1, row=2, sticky=E)
ttk.Spinbox(thresh_pane, from_=0, to=255, increment=1, width=5, textvariable=channel_thresh).grid(column=1, row=3, sticky=E)

def set_threshold(*args):
	update_thresh(select_channel.curselection()[0])
	
def update_thresh(channel):
	thresh_tracker[channel] = channel_thresh.get()
	image_dict['threshold'] = image_dict['color_image'][:,:,channel] >= channel_thresh.get()

def show_channel(*args):
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['color_image'][:,:,select_channel.curselection()[0]]))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	
def show_threshold(*args):
	image_dict['image1'] = ImageTk.PhotoImage(image = Image.fromarray(image_dict['threshold']))
	canvas.create_image(0,0,image=image_dict['image1'],anchor="nw")
	
ttk.Button(thresh_pane, text='Set threshold', command=set_threshold).grid(column=1, row=4, sticky=(W,E))
#################### SET THRESHOLD ####################

#################### SHOW CHANNELS #v4####################	
channel_show = StringVar(value=channel_list)
channel_select = Listbox(channel_pane, listvariable=channel_show, activestyle='none',
					 selectmode='extended', exportselection=0, width=10, height=len(channel_list))
channel_select.grid(column=1, row=1, sticky=(W,E))
channel_select.bind("<<ListboxSelect>>", divide_channels)
#################### SHOW CHANNELS #v4#####################

########################## ROI #############################
def set_color_str(*args):
	roi_colorstr.set(roi_colorkeys[select_roi.get()])
	
roi_colorstr = StringVar()
select_roi = StringVar()
ttk.OptionMenu(roi_pane, select_roi, ROIs[0], *ROIs, command=set_color_str).grid(column=1, row=1, sticky=E)

roi_entry = ttk.Entry(roi_pane, textvariable=roi_colorstr, width=14)
roi_entry.grid(column=2, row=1, sticky=W)
roi_entry.config(state="disabled")

ttk.Button(roi_pane, text='Draw polygon', command=idraw_polygon).grid(column=1, row=2, sticky=(W,E))
ttk.Button(roi_pane, text='Undraw polygon', command=undraw_polygon).grid(column=2, row=2, sticky=(W,E))

ttk.Button(roi_pane, text='Draw rectangle', command=idraw_rectangle).grid(column=1, row=3, sticky=(W,E))
ttk.Button(roi_pane, text='Undraw rectangle', command=undraw_rectangle).grid(column=2, row=3, sticky=(W,E))

ttk.Button(roi_pane, text='Draw line', command=idraw_line).grid(column=1, row=4, sticky=(W,E))
ttk.Button(roi_pane, text='Undraw line', command=undraw_line).grid(column=2, row=4, sticky=(W,E))

ttk.Button(roi_pane, text='Reset ROI', command=reset_polygon).grid(column=1, row=5, sticky=(W,E))
ttk.Button(roi_pane, text='Delete all ROIs', command=delete_rois).grid(column=2, row=5, sticky=(W,E))

ttk.Button(roi_pane, text='Save ROIs', command=save_rois).grid(column=1, row=6, sticky=(W,E))
############################ ROI ###########################

###########################v3 masks ############################
# Show masks
def update_display(*args):
	switches['display_masks'] = show_masks.curselection()
	update_mask()
choicesvar = StringVar(value=LABELS)
show_masks = Listbox(masks_pane, listvariable=choicesvar, activestyle='none',
					 selectmode='extended', exportselection=0, width=15, height=len(LABELS))
show_masks.select_set(3)
show_masks.grid(column=1, row=1, sticky=(W,E))
show_masks.bind("<<ListboxSelect>>", update_display)

def hide_all(*args):
	show_masks.selection_clear(0, END)
	switches.pop('display_masks', None) # really should be an empty tuple, but doesn't matter
	update_mask()
ttk.Button(masks_pane, text='Hide all', command=hide_all).grid(column=1, row=2, sticky=(W,E))

ttk.Button(masks_pane, text='Save Mask', command=save_masks).grid(column=1, row=3, sticky=(W,E))

def update_flip(*args):
	switches['flip_from'] = from_.curselection()
	switches['flip_to'] = to_.curselection()
	select_polygon() # v14
choicesvar = StringVar(value=LABELS)

ttk.Label(toolbar_pane, text='Flip from:').grid(column=1, row=1, sticky=(W,E))
from_ = Listbox(toolbar_pane, listvariable=choicesvar, selectmode='extended', activestyle='none',
				exportselection=0, width=15, height=len(LABELS))
from_.grid(column=1, row=2, sticky=(W,E))
from_.bind("<<ListboxSelect>>", update_flip)

ttk.Label(toolbar_pane, text='Flip to:').grid(column=2, row=1, sticky=(W,E))
to_ = Listbox(toolbar_pane, listvariable=choicesvar, selectmode='browse', activestyle='none',
			  exportselection=0, width=15, height=len(LABELS))
to_.grid(column=2, row=2, sticky=(W,E))
to_.bind("<<ListboxSelect>>", update_flip)

ttk.Button(toolbar_pane, text='Polygon flip', command=select_polygon).grid(column=1, row=3, sticky=(W,E))
ttk.Button(toolbar_pane, text='Reset polygon', command=reset_polygon).grid(column=2, row=3, sticky=(W,E))
ttk.Button(toolbar_pane, text='Undo', command=undo_manip).grid(column=1, row=4, sticky=(W,E))

def reset_mask_button(*args):
	divide_masks(image_dict['mask_image'])
	update_mask(save_temp=True)
	
ttk.Button(toolbar_pane, text='Reset mask', command=reset_mask_button).grid(column=2, row=4, sticky=(W,E))

# v11
dot_radius = IntVar()
dot_radius.set(4)
ttk.Button(dot_pane, text='Draw dot', command=dot_ec).grid(column=1, row=1, sticky=(W,E))
ttk.Label(dot_pane, text='Dot radius:').grid(column=2, row=1, sticky=E)
ttk.Spinbox(dot_pane, from_=1, to=10, increment=1, width=3, textvariable=dot_radius, 
			command=dot_ec).grid(column=3, row=1, sticky=E) # v14

###########################v3 masks ############################

#################### INFORMATION ######################
# About button
def about():
	messagebox.showinfo(
		message='GUI for selecting ROIs. Version '+version_no
	)
ttk.Button(about_pane, text='About', command=about).grid(column=1, row=1, sticky=(W,E))

def hot_keys(*args):
	new_window = Toplevel(root)
	new_window.title('Hot Keys')
	new_window.resizable(False, False)
	topframe = ttk.Panedwindow(new_window, orient=VERTICAL)
	topframe.pack(fill=BOTH, expand=1)
	topframe.grid_columnconfigure(0, weight=1) # expand frame to fill extra space if window is resized
	topframe.grid_rowconfigure(0, weight=1)
	
	report = Text(topframe, width=50, height=6, background='white')
	report.grid(column=1, row=1, sticky=(W,E))
	report.insert('1.0',
'''tab: open image
hold left shift: display original image without ROI overlay
hold control: display dapi image (if exists in a subdirectory called "dapi")
hold left alt: display single channel image
hold right alt: display thresholded channel image
right shift: save ROIs
''')
	report.config(state=DISABLED)
ttk.Button(about_pane, text='Hot keys', command=hot_keys).grid(column=1, row=2, sticky=(W,E))

def helping():
	new_window = Toplevel(root)
	new_window.title('Help')
	new_window.resizable(False, False)
	topframe = ttk.Panedwindow(new_window, orient=VERTICAL)
	topframe.pack(fill=BOTH, expand=1)
	topframe.grid_columnconfigure(0, weight=1) # expand frame to fill extra space if window is resized
	topframe.grid_rowconfigure(0, weight=1)
	
	report = Text(topframe, width=200, height=44, background='white')
	report.grid(column=1, row=1, sticky=(W,E))
	report.insert('1.0', 
'''## Image acquisition
Open image: opens windows dialog box for selecting image to open, then opens another windows dialog box for selecting the mask (labels) file to open. If "Mass analysis" mode is checked, will open images (and corresponding masks) from "Image directory" (and "Mask directory") sequentially each time "Open image" is clicked. If there is an updated version of the mask (filename starts with "updated_"), it will be preferentially loaded. Also loads the original image file in the working directory; if subdirectory "Original" exists, will load the image file in that subdirectory with the closest name to the current image file.

Save ROIs: creates a subdirectory in the specified directory called "ROIs", and pickles each ROI category in a file containing a list of tuples containing type (polygon, rectangle, line) and coordinates, as in (type, coordinates)

Mass analysis: check to enter mass analysis mode ("Open image" will open image files in the directory sequentially).

Continue: if both "Mass analysis" and "Continue" are checked, "Open image" will open images that do not have corresponding ROIs in the ROI directory.

''')
	report.config(state=DISABLED)
ttk.Button(about_pane, text='Help', command=helping).grid(column=1, row=3, sticky=(W,E))

def support():
	messagebox.showinfo(message='Please report bugs to Lu Yang at yanglum at stanford.edu')
ttk.Button(about_pane, text='Support', command=support).grid(column=1, row=4, sticky=(W,E))
#################### INFORMATION ######################

root.bind("<Tab>", open_file)
root.bind("<Shift_L>", show_roi)
root.bind("<KeyRelease-Shift_L>", show_mask_image)
root.bind("<Control_L>", show_dapi_image)
root.bind("<KeyRelease-Control_L>", show_mask_image)
root.bind("<Shift_R>", save_masks)
root.bind("<Alt_L>", show_mask_image) # v3
#root.bind("<KeyRelease-Alt_L>", show_mask_image) # v3
root.bind("<Alt_R>", show_threshold) # v3
root.bind("<KeyRelease-Alt_R>", show_channel) # v3

# add padding around each widget
for child in mainframe.winfo_children():
	child.grid_configure(padx=10, pady=10)
	
root.mainloop()


# In[ ]:




