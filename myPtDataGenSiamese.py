import os
import json
import math
import numpy as np 
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from skimage.transform import resize
import random

class Pt_datagen_siamese:

	def __init__(self,data_dir,anno_dir,model_input_shape,model_output_shape,batch_size_select,data_for):


		self.id_to_file_dict = {}
		self.vid_to_id_dict = {}
		self.id_to_track_id = {}

		# self.id_to_kpv = {}
		# self.id_to_kp = {}
		# self.id_to_valid = {}
		# self.id_to_wh = {}

		self.pair_dict = {}

		self.start_idx = []
		self.end_idx = []

		self.img_ids = []

		self.input_shape = model_input_shape
		self.output_shape = model_output_shape

		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None

		self.data_dir = data_dir
		self.anno_dir = anno_dir
		self.data_for = data_for

		self.get_data_from_dir()
		self.get_pair()
		# self.start_idx, self.end_idx, self.n_batchs = self.get_start_end_idx()
		# self.split_kp_and_v()

		# self.limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
		# self.n_keypoints = 15
		# self.n_limbs = 16
		print('Create datagen_{}: Done ...'.format(self.data_for))


	def get_data_from_dir(self):
		temp_image_id_with_label = []
		temp_file_name_with_label = []
		temp_anno_bbox_list = []
		temp_anno_track_id_list = []
		temp_anno_id_list = []
		temp_anno_kp_list = []
		temp_vid_to_id_dict = {}
		temp_id_to_track_id = {}
		temp_id_to_kpv = {}
		temp_id_to_bbox = {}

		temp_anno_dir = self.anno_dir + self.data_for + '/'
		for anno_file in os.listdir(temp_anno_dir):
			current_image_id_with_label = []
			if anno_file.endswith('.json'):
				temp = temp_anno_dir + anno_file
				with open(temp) as f:
					data = json.load(f)
				data_images = data['images']
				data_annotations = data['annotations']
			for temp_image in data_images:
				if temp_image['is_labeled']:
					temp_image_id_with_label.append(temp_image['id'])
					current_image_id_with_label.append(temp_image['id'])
					temp_file_name_with_label.append(temp_image['file_name'])
			temp_vid_to_id_dict[anno_file] = current_image_id_with_label

			
			current_track_id = []
			current_image_id = []
			# current_kpv = []
			for anno in data_annotations:
				
				temp_keys = list(anno.keys())
				to_check_keys = ['image_id','bbox','track_id']
				if all(item in temp_keys for item in to_check_keys):
					if anno['image_id'] in temp_image_id_with_label:
						bbox_temp = anno['bbox']
						if bbox_temp[2] > 0 and bbox_temp[3] > 0:
							if bbox_temp[0] >= 0 and bbox_temp[1] >= 0:
								temp_anno_bbox_list.append(anno['bbox'])
								current_track_id.append(anno['track_id'])
								current_image_id.append(anno['image_id'])
								# current_kpv.append(anno['keypoints'])

				# create dict

			current_unique_image_id = list(set(current_image_id))
			for cid in current_unique_image_id:
				cidx = [ x for x in range(len(current_image_id)) if current_image_id[x] == cid]
				c_track_id = [current_track_id[x] for x in cidx]
				c_bbox = [temp_anno_bbox_list[x] for x in cidx]
				# c_kpv = [current_kpv[x] for x in cidx]

				temp_id_to_track_id[cid] = c_track_id
				temp_id_to_bbox[cid] = c_bbox
				# temp_id_to_kpv[cid] = c_kpv


		temp_id_to_file_dict = {temp_image_id_with_label[i]:temp_file_name_with_label[i] for i in range(len(temp_image_id_with_label))}

		self.id_to_file_dict = temp_id_to_file_dict
		self.vid_to_id_dict = temp_vid_to_id_dict
		self.id_to_track_id = temp_id_to_track_id
		self.id_to_bbox_dict = temp_id_to_bbox
		# self.id_to_kpv = temp_id_to_kpv

	def scale_43(self,bboxes,img_w,img_h):
		out_bboxes = []
		for i in range(len(bboxes)):
			i_bbox = bboxes[i]
			print(i_bbox)
			bbox_x, bbox_y,bbox_w,bbox_h = i_bbox
			# bbox_x2 = bbox_x+bbox_w
			# bbox_y2 = bbox_y+bbox_h
			# bbox_w = bbox_x2-bbox_x
			# bbox_h = bbox_y2-bbox_y
			# print('Before: {}, {}, {}, {}'.format(bbox_x,bbox_y,bbox_w,bbox_h))
			to_check = 0.75*bbox_h
			if to_check >= bbox_w:
				add_x = True
				# print('Add x')
			else:
				add_x = False
				# print('Add y')


			if add_x:
				new_bbox_h = bbox_h
				new_bbox_w = 0.75*bbox_h
				diff = new_bbox_w - bbox_w
				new_bbox_y = bbox_y
				new_bbox_x = bbox_x - 0.5*diff
				#check if in image
				if new_bbox_x < 0:
					new_bbox_x = 0
				if new_bbox_x+new_bbox_w >= img_w:
					new_bbox_x = img_w - new_bbox_w - 1
			else:
				new_bbox_w = bbox_w
				new_bbox_h = 4.0/3.0 * bbox_w
				diff = new_bbox_h - bbox_h
				new_bbox_x = bbox_x
				new_bbox_y = bbox_y - 0.5 * diff

				if new_bbox_y < 0:
					new_bbox_y = 0
				if new_bbox_y+new_bbox_h >= img_h:
					new_bbox_y = img_h - new_bbox_h - 1
			temp_new_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]
			# print('After: {}, {}, {}, {}'.format(new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h))
			out_bboxes.append(temp_new_bbox)
		return out_bboxes

	def get_pair(self):
		temp_vid_to_id_dict = self.vid_to_id_dict
		temp_id_to_track_id = self.id_to_track_id
		temp_id_to_bbox_dict = self.id_to_bbox_dict
		# temp_valid_keys = self.id_to_kpv
		temp_imgA_id = []
		temp_imgB_id = []
		temp_imgA_bbox = []
		temp_imgB_boxx = []
		temp_y = []
		temp_img_ids = []

		for vid,i_ids in temp_vid_to_id_dict.items():
			len_imgs_in_vid = len(i_ids)
			temp_first_img = i_ids[0]
			temp_img = io.imread(self.data_dir + self.id_to_file_dict[temp_first_img])
			img_h = temp_img.shape[0]
			img_w = temp_img.shape[1]

			for i in range(len_imgs_in_vid-1):
				idA = i_ids[i]
				idB = i_ids[i+1]
				bboxesA = temp_id_to_bbox_dict[idA]
				bboxesB = temp_id_to_bbox_dict[idB]
				bboxesA = self.scale_43(bboxesA,img_w,img_h)
				bboxesB = self.scale_43(bboxesB,img_w,img_h)
				tracksA = temp_id_to_track_id[idA]
				tracksB = temp_id_to_track_id[idB]
				for iA in range(len(tracksA)):
					bboxA  = bboxesA[iA]
					trackA = tracksA[iA]
					
					for iB in range(len(tracksB)):
						bboxB = bboxesB[iB]
						trackB = tracksB[iB]
						

						temp_imgA_id.append(idA)
						temp_imgB_id.append(idB)
						temp_imgA_bbox.append(bboxA)
						temp_imgB_bbox.append(bboxB)

						if trackA == trackB:
							temp_y.append(1)
						else:
							temp_y.append(0)

		self.imgA_id = temp_imgA_id
		self.imgA_bbox = temp_imgA_bbox
		self.imgB_id = temp_imgB_id
		self.imgB_bbox = temp_imgB_bbox
		self.labels = temp_y
		self.n_imgs = len(temp_imgA_id)

	def get_start_end_idx(self):
		max_idx = self.n_imgs
		temp_batch_size = self.batch_size
		l = list(range(max_idx))
		temp_start_idx = l[0::temp_batch_size]
		def add_batch_size(num,max_id=max_idx,bz=temp_batch_size):
			return min(num+bz,max_id)
		temp_end_idx = list(map(add_batch_size,temp_start_idx))
		temp_n_batchs = len(temp_start_idx)

		return temp_start_idx, temp_end_idx, temp_n_batchs

	# def get_target_valid_joint(self,input_kav):
	# 	splited_kps = []
	# 	splited_valids = []
	# 	for temp_anno_kp in input_kav:
	# 		temp_x = np.array(temp_anno_kp[0::3])
	# 		temp_y = np.array(temp_anno_kp[1::3])
	# 		temp_valid = np.array(temp_anno_kp[2::3])
	# 		temp_valid = temp_valid > 0
	# 		temp_valid = temp_valid.astype('float32')
	# 		temp_target_coord = np.stack([temp_x,temp_y],axis=1)
	# 		temp_target_coord = temp_target_coord.astype('float32')

	# 		splited_kps.append(temp_target_coord)
	# 		splited_valids.append(temp_valid)

	# 	return splited_kps,splited_valids

	# def split_kp_and_v(self):
	# 	temp_id_to_kp = {}
	# 	temp_id_to_valid = {}
	# 	temp_id_to_kpv = self.id_to_kpv
	# 	for i_id,kpv in temp_id_to_kpv.items():
	# 		t_ks,t_vs = self.get_target_valid_joint(kpv)
	# 		for i in range(len(t_ks)):
	# 			temp_k = t_ks[i]
	# 			temp_v = t_vs[i]

	# 			t_ks[i] = np.delete(temp_k,[1,2],0)
	# 			t_vs[i] = np.delete(temp_v,[1,2])

	# 		temp_id_to_kp[i_id] = t_ks
	# 		temp_id_to_valid[i_id] = t_vs

	# 	self.id_to_kp = temp_id_to_kp
	# 	self.id_to_valid = temp_id_to_valid

	# def render_tff(self,i_grid_x,i_grid_y,i_kp_A,i_kp_B,i_v_A,i_v_B,accumulate_vec_map,i_h,i_w):
	# 	if (i_v_A != 0) and (i_v_B !=0):
	# 		i_kp_A = i_kp_A.astype('float')
	# 		i_kp_B = i_kp_B.astype('float')
	# 		thre = 1.2  # limb width

	# 		centerA = (i_kp_A[0] * i_grid_x / i_w,i_kp_A[1] * i_grid_y / i_h)
	# 		centerB = (i_kp_B[0] * i_grid_x / i_w,i_kp_B[1] * i_grid_y / i_h)

	# 		limb_vec = i_kp_B - i_kp_A
	# 		norm = np.linalg.norm(limb_vec)
	# 		if (norm == 0.0):
	# 			# print('limb is too short, ignore it...')
	# 			return accumulate_vec_map

	# 		limb_vec_unit = limb_vec / norm

	# 		min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
	# 		max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), i_grid_x)
	# 		min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
	# 		max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), i_grid_y)
	# 		if min_x >= max_x:
	# 			min_x = max_x - 1
	# 		if min_y >= max_y:
	# 			min_y = max_y - 1
	# 		# print('min_x: {}'.format(min_x))
	# 		# print('max_x: {}'.format(max_x))
	# 		# print('min_y: {}'.format(min_y))
	# 		# print('max_y: {}'.format(max_y))
	# 		range_x = list(range(int(min_x), int(max_x), 1))
	# 		range_y = list(range(int(min_y), int(max_y), 1))
	# 		# print('range_x: {}'.format(range_x))
	# 		# print('range_y: {}'.format(range_y))
	# 		xx, yy = np.meshgrid(range_x, range_y)
	# 		ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
	# 		ba_y = yy - centerA[1]
	# 		limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
	# 		mask = limb_width < thre  # mask is 2D

	# 		vec_map = np.copy(accumulate_vec_map) * 0.0
	# 		vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
	# 		vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

	# 		mask = np.logical_or.reduce((np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))
	# 		accumulate_vec_map += vec_map

	# 		return accumulate_vec_map
	# 	else:
	# 		# print('not valid')
	# 		return accumulate_vec_map

	# def render_heatmap(self,i_grid_x,i_grid_y,i_kp,i_v,sigma,accumulate_confid_map,i_h,i_w):
	# 	if i_v != 0:
	# 		y_range = [i for i in range(int(i_grid_y))]
	# 		x_range = [i for i in range(int(i_grid_x))]
	# 		xx, yy = np.meshgrid(x_range,y_range)
	# 		t_x = i_kp[0] * i_grid_x / i_w
	# 		t_y = i_kp[1] * i_grid_y / i_h
	# 		# print(t_kp)
	# 		d2 = (xx - t_x)**2 + (yy - t_y)**2
	# 		exponent = d2 / 2.0 / sigma / sigma
	# 		mask = exponent <= 4.6052
	# 		cofid_map = np.exp(-exponent)
	# 		cofid_map = np.multiply(mask, cofid_map)
	# 		accumulate_confid_map += cofid_map
	# 		accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
	# 		return accumulate_confid_map
	# 	else:
	# 		return accumulate_confid_map

	# def render_paf(self,i_grid_x,i_grid_y,i_kp_A,i_kp_B,i_v_A,i_v_B,accumulate_vec_map,i_h,i_w):
	# 	if (i_v_A != 0) and (i_v_B !=0):
	# 		i_kp_A = i_kp_A.astype('float')
	# 		i_kp_B = i_kp_B.astype('float')
	# 		#1.25
	# 		thre = 1  # limb width

	# 		centerA = (i_kp_A[0] * i_grid_x / i_w,i_kp_A[1] * i_grid_y / i_h)
	# 		centerB = (i_kp_B[0] * i_grid_x / i_w,i_kp_B[1] * i_grid_y / i_h)

	# 		limb_vec = i_kp_B - i_kp_A
	# 		norm = np.linalg.norm(limb_vec)
	# 		if (norm == 0.0):
	# 			# print('limb is too short, ignore it...')
	# 			return accumulate_vec_map

	# 		limb_vec_unit = limb_vec / norm

	# 		min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
	# 		max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), i_grid_x)
	# 		min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
	# 		max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), i_grid_y)
	# 		if min_x >= max_x:
	# 			min_x = max_x - 1
	# 		if min_y >= max_y:
	# 			min_y = max_y - 1
	# 		range_x = list(range(int(min_x), int(max_x), 1))
	# 		range_y = list(range(int(min_y), int(max_y), 1))
	# 		if len(range_x) == 0 or len(range_y) == 0:
	# 			print('there is something wrong')
	# 			print('range_x: {}'.format(range_x))
	# 			print('range_y: {}'.format(range_y))

	# 		xx, yy = np.meshgrid(range_x, range_y)
	# 		ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
	# 		ba_y = yy - centerA[1]
	# 		limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
	# 		mask = limb_width < thre  # mask is 2D

	# 		vec_map = np.copy(accumulate_vec_map) * 0.0
	# 		vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
	# 		vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

	# 		mask = np.logical_or.reduce((np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))
	# 		accumulate_vec_map += vec_map

	# 		return accumulate_vec_map
	# 	else:
	# 		# print('not valid')
	# 		return accumulate_vec_map

	# def gen_batch(self,batch_order):
	# 	batch_imgs = []
	# 	batch_heatmaps = []
	# 	batch_pafs = []
	# 	batch_valids = []
	# 	b_start = self.start_idx[batch_order]
	# 	b_end = self.end_idx[batch_order]
	# 	temp_output_shape = self.output_shape
	# 	temp_input_shape = self.input_shape

	# 	dict_id_to_file = self.id_to_file_dict
	# 	dict_id_to_kp = self.id_to_kp
	# 	dict_id_to_valid = self.id_to_valid
	# 	dict_id_to_wh = self.id_to_wh

	# 	temp_img_dir = self.data_dir
	# 	temp_img_ids = self.img_ids

	# 	for idx in range(b_start,b_end):
	# 		channels_heat = self.n_keypoints
	# 		channels_paf = 2*self.n_limbs

	# 		id_0 = temp_img_ids[idx]

	# 		temp_wh = dict_id_to_wh[id_0]
	# 		temp_w = temp_wh[0]
	# 		temp_h = temp_wh[1]

	# 		kps_0 = dict_id_to_kp[id_0]
	# 		valid_0 = dict_id_to_valid[id_0]

	# 		grid_y = self.output_shape[1]
	# 		grid_x = self.output_shape[0]

	# 		# img
	# 		img_0 = io.imread(self.data_dir + self.id_to_file_dict[id_0])
	# 		img_0 = resize(img_0,temp_input_shape).astype('float32')

	# 		# heatmap
	# 		heatmaps = np.zeros((int(grid_y),int(grid_x),channels_heat))
	# 		for i in range(len(kps_0)):
	# 			for j in range(channels_heat):
	# 				heatmaps[:,:,j] = self.render_heatmap(grid_x,grid_y,kps_0[i][j],valid_0[i][j],2,heatmaps[:,:,j],temp_h,temp_w)

	# 		# paf
	# 		pafs = np.zeros((int(grid_y),int(grid_x),channels_paf))
	# 		temp_limb_list = self.limb_list
	# 		for i in range(len(kps_0)):
	# 			for j in range(len(temp_limb_list)):
	# 				limb = temp_limb_list[j]
	# 				idx_A = limb[0]
	# 				idx_B = limb[1]
	# 				pafs[:,:,[j,j+self.n_limbs]] = self.render_paf(grid_x,grid_y,kps_0[i][idx_A],kps_0[i][idx_B],valid_0[i][idx_A],valid_0[i][idx_B],pafs[:,:,[j,j+self.n_limbs]],temp_h,temp_w)

	# 		if len(img_0.shape) > 2:
	# 			batch_imgs.append(img_0)
	# 			batch_heatmaps.append(heatmaps)
	# 			batch_valids.append(valid_0)
	# 			batch_pafs.append(pafs)

	# 	batch_imgs = np.array(batch_imgs)

	# 	batch_heatmaps = np.array(batch_heatmaps)
	# 	batch_pafs = np.array(batch_pafs)

	# 	return batch_imgs, batch_heatmaps, batch_pafs, batch_valids



	# def gen_batch_tff(self,batch_order):
	# 	batch_imgs_0 = []
	# 	batch_imgs_1 = []

	# 	batch_TFF = []
	# 	batch_valids_0 = []
	# 	batch_valids_1 = []
	# 	b_start = self.start_idx[batch_order]
	# 	b_end = self.end_idx[batch_order]
	# 	temp_output_shape = self.output_shape
	# 	temp_input_shape = self.input_shape

	# 	dict_id_to_file = self.id_to_file_dict
	# 	dict_id_to_track_id = self.id_to_track_id
	# 	dict_id_to_kp = self.id_to_kp
	# 	dict_id_to_valid = self.id_to_valid
	# 	dict_id_to_wh = self.id_to_wh
	# 	dict_pair = self.pair_dict

	# 	temp_img_dir = self.data_dir
	# 	temp_img_ids = self.img_ids

	# 	for idx in range(b_start,b_end):
	# 		channels_tff = self.n_keypoints * 2

	# 		# get data

	# 		id_0 = temp_img_ids[idx]

	# 		id_1 = dict_pair[id_0]

	# 		temp_wh = dict_id_to_wh[id_0]
	# 		temp_w = temp_wh[0]
	# 		temp_h = temp_wh[1]

	# 		kps_0 = dict_id_to_kp[id_0]
	# 		kps_1 = dict_id_to_kp[id_1]

	# 		valid_0 = dict_id_to_valid[id_0]
	# 		valid_1 = dict_id_to_valid[id_1]

	# 		track_id_0 = dict_id_to_track_id[id_0]
	# 		track_id_1 = dict_id_to_track_id[id_1]

	# 		grid_y = self.output_shape[1]
	# 		grid_x = self.output_shape[0]

	# 		# load img
	# 		img_0 = io.imread(self.data_dir + self.id_to_file_dict[id_0])
	# 		img_1 = io.imread(self.data_dir + self.id_to_file_dict[id_1])

	# 		img_0 = resize(img_0,temp_input_shape).astype('float32')
	# 		img_1 = resize(img_1,temp_input_shape).astype('float32')

	# 		# TFF
	# 		tffs = np.zeros((int(grid_y),int(grid_x),channels_tff))
	# 		mutual_track_id = list(set(track_id_0).intersection(track_id_1))
	# 		tracked_0_idx = [x for x in range(len(track_id_0)) if track_id_0[x] in mutual_track_id]
	# 		tracked_1_idx = [x for x in range(len(track_id_1)) if track_id_1[x] in mutual_track_id]

	# 		if len(tracked_0_idx) != len(tracked_1_idx):
	# 			print(track_id_0)
	# 			print(track_id_1)
	# 			print('Something wrong')
	# 			print(mutual_track_id)
	# 			print(tracked_0_idx)
	# 			print(tracked_1_idx)

	# 		for i in range(len(tracked_0_idx)):
	# 			idx_0 = tracked_0_idx[i]
	# 			idx_1 = tracked_1_idx[i]
	# 			tkp_0 = kps_0[idx_0]
	# 			tkp_1 = kps_1[idx_1]
	# 			tv_0 = valid_0[idx_0]
	# 			tv_1 = valid_1[idx_1]
	# 			for j in range(self.n_keypoints):
	# 				# for debug
	# 				# print('idx: {}'.format(idx))
	# 				# print('i: {}'.format(i))
	# 				# print('j: {}'.format(j))

	# 				tffs[:,:,[j,j+self.n_keypoints]] = self.render_tff(grid_x,grid_y,tkp_0[j],tkp_1[j],tv_0[j],tv_1[j],tffs[:,:,[j,j+self.n_keypoints]],temp_h,temp_w)

	# 		if len(img_0.shape) > 2 and len(img_1.shape) >2:
	# 			# concat imgs
	# 			batch_imgs_0.append(img_0)
	# 			batch_imgs_1.append(img_1)

	# 			# concat valids
	# 			batch_valids_0.append(valid_0)
	# 			batch_valids_1.append(valid_1)

	# 			#concat TFFs
	# 			batch_TFF.append(tffs)
	# 	batch_imgs_0 = np.array(batch_imgs_0)
	# 	batch_imgs_1 = np.array(batch_imgs_1)

	# 	batch_TFF = np.array(batch_TFF)

	# 	return batch_imgs_0,batch_imgs_1, batch_TFF, batch_valids_0,batch_valids_1
	# def shuffle_order(self):
	# 	temp_img_ids = self.img_ids
	# 	random.shuffle(temp_img_ids)
	# 	self.img_ids = temp_img_ids