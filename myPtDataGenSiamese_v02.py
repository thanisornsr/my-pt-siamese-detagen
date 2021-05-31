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
		self.start_idx, self.end_idx, self.n_batchs = self.get_start_end_idx()
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
			current_kpv = []
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
								# temp_anno_kp_list.append(anno['keypoints'])
								current_kpv.append(anno['keypoints'])

				# create dict

			current_unique_image_id = list(set(current_image_id))
			for cid in current_unique_image_id:
				cidx = [ x for x in range(len(current_image_id)) if current_image_id[x] == cid]
				c_track_id = [current_track_id[x] for x in cidx]
				c_bbox = [temp_anno_bbox_list[x] for x in cidx]
				c_kpv = [current_kpv[x] for x in cidx]

				temp_id_to_track_id[cid] = c_track_id
				temp_id_to_bbox[cid] = c_bbox
				temp_id_to_kpv[cid] = c_kpv


		temp_id_to_file_dict = {temp_image_id_with_label[i]:temp_file_name_with_label[i] for i in range(len(temp_image_id_with_label))}

		self.id_to_file_dict = temp_id_to_file_dict
		self.vid_to_id_dict = temp_vid_to_id_dict
		self.id_to_track_id = temp_id_to_track_id
		self.id_to_bbox_dict = temp_id_to_bbox
		self.unique_id = current_unique_image_id
		self.id_to_kpv = temp_id_to_kpv

	def scale_43(self,bboxes,img_w,img_h):
		out_bboxes = []
		for i in range(len(bboxes)):
			i_bbox = bboxes[i]
			# print(i_bbox)
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
		temp_id_to_kpv = self.id_to_kpv
		# temp_valid_keys = self.id_to_kpv
		temp_imgA_id = []
		temp_imgB_id = []
		temp_imgA_bbox = []
		temp_imgB_bbox = []
		temp_imgA_kp = []
		temp_imgB_kp = []
		temp_y = []
		temp_img_ids = []
		# temp_imgA = []
		# temp_imgB = []

		for vid,i_ids in temp_vid_to_id_dict.items():
			len_imgs_in_vid = len(i_ids)
			temp_first_img = i_ids[0]
			temp_img = io.imread(self.data_dir + self.id_to_file_dict[temp_first_img])
			img_h = temp_img.shape[0]
			img_w = temp_img.shape[1]

			for i in range(len_imgs_in_vid-1):
				idA = i_ids[i]
				idB = i_ids[i+1]
				if idA not in temp_id_to_bbox_dict or idB not in temp_id_to_bbox_dict:
					continue
				bboxesA = temp_id_to_bbox_dict[idA]
				bboxesB = temp_id_to_bbox_dict[idB]
				bboxesA = self.scale_43(bboxesA,img_w,img_h)
				bboxesB = self.scale_43(bboxesB,img_w,img_h)
				kpvsA = temp_id_to_kpv[idA]
				kpvsB = temp_id_to_kpv[idB]

				kpsA,vsA = self.get_target_valid_joint(kpvsA)
				kpsB,vsB = self.get_target_valid_joint(kpvsB)



				tracksA = temp_id_to_track_id[idA]
				tracksB = temp_id_to_track_id[idB]
				for iA in range(len(tracksA)):
					bboxA  = bboxesA[iA]
					trackA = tracksA[iA]
					kpA = kpsA[iA]
					
					for iB in range(len(tracksB)):
						bboxB = bboxesB[iB]
						trackB = tracksB[iB]
						kpB = kpsB[iB]

						kpA = np.delete(kpA,[1,2],0)
						kpB = np.delete(kpB,[1,2],0)

						
						temp_imgA_id.append(idA)
						temp_imgB_id.append(idB)
						temp_imgA_bbox.append(bboxA)
						temp_imgB_bbox.append(bboxB)
						temp_imgA_kp.append(kpA)
						temp_imgB_kp.append(kpB)


						if trackA == trackB:
							temp_y.append(1)
						else:
							temp_y.append(0)

		self.imgA_id = temp_imgA_id
		self.imgA_bbox = temp_imgA_bbox
		# self.imgA = temp_imgA
		self.imgB_id = temp_imgB_id
		# self.imgB = temp_imgB
		self.imgB_bbox = temp_imgB_bbox
		
		self.imgA_kp = temp_imgA_kp
		self.imgB_kp = temp_imgB_kp

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

	def get_target_valid_joint(self,input_kav):
		splited_kps = []
		splited_valids = []
		for temp_anno_kp in input_kav:
			temp_x = np.array(temp_anno_kp[0::3])
			temp_y = np.array(temp_anno_kp[1::3])
			temp_valid = np.array(temp_anno_kp[2::3])
			temp_valid = temp_valid > 0
			temp_valid = temp_valid.astype('float32')
			temp_target_coord = np.stack([temp_x,temp_y],axis=1)
			temp_target_coord = temp_target_coord.astype('float32')

			splited_kps.append(temp_target_coord)
			splited_valids.append(temp_valid)

		return splited_kps,splited_valids

	def split_kp_and_v(self):
		temp_id_to_kp = {}
		temp_id_to_valid = {}
		temp_id_to_kpv = self.id_to_kpv
		for i_id,kpv in temp_id_to_kpv.items():
			t_ks,t_vs = self.get_target_valid_joint(kpv)
			for i in range(len(t_ks)):
				temp_k = t_ks[i]
				temp_v = t_vs[i]

				t_ks[i] = np.delete(temp_k,[1,2],0)
				t_vs[i] = np.delete(temp_v,[1,2])

			temp_id_to_kp[i_id] = t_ks
			temp_id_to_valid[i_id] = t_vs

		self.id_to_kp = temp_id_to_kp
		self.id_to_valid = temp_id_to_valid

	def gen_batch_kp_bbox(self,batch_order):
		batch_kp_A = []
		batch_kp_B = []
		batch_bbox_A = []
		batch_bbox_B = []
		batch_y = []

		temp_input_shape = self.input_shape

		b_start = self.start_idx[batch_order]
		b_end = self.end_idx[batch_order]

		temp_imgsA = self.imgA_id
		temp_imgsA_bbox = self.imgA_bbox
		temp_imgsA_kp = self.imgA_kp

		temp_imgsB = self.imgB_id
		temp_imgsB_bbox = self.imgB_bbox
		temp_imgsB_bbox = self.imgB_kp

		temp_y = self.labels

		temp_dict = self.id_to_file_dict

		for i in range(b_start,b_end):
			#imgA
			img_id = temp_imgsA[i]
			# i_dir = temp_dict[img_id]
			# i_dir = self.data_dir + i_dir
			# o_img = mpimg.imread(i_dir)
			# if o_img.shape[0] == 0 or o_img.shape[1]  == 0 or o_img.ndim < 3:
			# 	continue
			A_bbox = temp_imgsA_bbox[i]
			A_kp = temp_imgsA_kp[i]
			# o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			# if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
			# 	# print('Detect empty image: '+i_dir)
			# 	continue
			# o_crop = resize(o_crop,temp_input_shape)
			# A_crop = o_crop.astype('float32')
			
			#imgB
			img_id = temp_imgsB[i]
			# i_dir = temp_dict[img_id]
			# i_dir = self.data_dir + i_dir
			# o_img = mpimg.imread(i_dir)
			# if o_img.shape[0] == 0 or o_img.shape[1]  == 0 or o_img.ndim < 3:
			# 	continue
			B_bbox = temp_imgsB_bbox[i]
			B_kp = temp_imgB_kp[i]
			# o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			# if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
			# 	# print('Detect empty image: '+i_dir)
			# 	continue
			# o_crop = resize(o_crop,temp_input_shape)
			# B_crop = o_crop.astype('float32')
			batch_bbox_A.append(A_bbox)
			batch_bbox_B.append(B_bbox)
			batch_kp_A.append(A_kp)
			batch_kp_B.append(B_kp)
			#y
			batch_y.append(temp_y[i])

		batch_bbox_A = np.array(batch_bbox_A)
		batch_bbox_B = np.array(batch_bbox_B)
		batch_kp_A = np.array(batch_kp_A)
		batch_kp_B = np.array(batch_kp_B)
		batch_y = np.array(batch_y)
		batch_y = batch_y.astype('float32')

		return batch_bbox_A,batch_bbox_B,batch_kp_A,batch_kp_B,batch_y


	def gen_batch(self,batch_order):
		batch_imgsA = []
		batch_imgsB = []
		batch_y = []

		temp_input_shape = self.input_shape

		b_start = self.start_idx[batch_order]
		b_end = self.end_idx[batch_order]
		temp_imgsA = self.imgA_id
		temp_imgsA_bbox = self.imgA_bbox
		temp_imgsB = self.imgB_id
		temp_imgsB_bbox = self.imgB_bbox
		temp_y = self.labels

		temp_dict = self.id_to_file_dict

		for i in range(b_start,b_end):

			#imgA
			img_id = temp_imgsA[i]
			i_dir = temp_dict[img_id]
			i_dir = self.data_dir + i_dir
			o_img = mpimg.imread(i_dir)
			if o_img.shape[0] == 0 or o_img.shape[1]  == 0 or o_img.ndim < 3:
				continue
			i_bbox = temp_imgsA_bbox[i]
			o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
				# print('Detect empty image: '+i_dir)
				continue
			o_crop = resize(o_crop,temp_input_shape)
			A_crop = o_crop.astype('float32')
			
			#imgB
			img_id = temp_imgsB[i]
			i_dir = temp_dict[img_id]
			i_dir = self.data_dir + i_dir
			o_img = mpimg.imread(i_dir)
			if o_img.shape[0] == 0 or o_img.shape[1]  == 0 or o_img.ndim < 3:
				continue
			i_bbox = temp_imgsB_bbox[i]
			o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			if o_crop.shape[0] == 0 or o_crop.shape[1]  == 0 or o_crop.shape[2] == 0:
				# print('Detect empty image: '+i_dir)
				continue
			o_crop = resize(o_crop,temp_input_shape)
			B_crop = o_crop.astype('float32')
			batch_imgsA.append(A_crop)
			batch_imgsB.append(B_crop)
			#y
			batch_y.append(temp_y[i])

		batch_imgsA = np.array(batch_imgsA)
		batch_imgsB = np.array(batch_imgsB)
		batch_y = np.array(batch_y)
		batch_y = batch_y.astype('float32')

		return batch_imgsA,batch_imgsB,batch_y

	def shuffle_order(self):
		temp_imgsA = self.imgA_id
		temp_imgsA_bbox = self.imgA_bbox
		temp_imgsA_kp = self.imgA_kp

		temp_imgsB = self.imgB_id
		temp_imgsB_bbox = self.imgB_bbox
		temp_imgsB_kp = self.imgB_kp

		temp_y = self.labels

		to_shuffle = list(zip(temp_imgsA,temp_imgsA_bbox,temp_imgsB,temp_imgsB_bbox,temp_y,temp_imgsA_kp,temp_imgsB_kp))
		random.shuffle(to_shuffle)
		temp_imgsA,temp_imgsA_bbox,temp_imgsB,temp_imgsB_bbox,temp_y,temp_imgsA_kp,temp_imgsB_kp = zip(*to_shuffle)


		self.imgA_id = temp_imgsA
		self.imgA_bbox = temp_imgsA_bbox
		self.imgA_kp = temp_imgsA_kp
		self.imgB_id = temp_imgsB
		self.imgB_bbox = temp_imgsB_bbox
		self.imgB_kp = temp_imgsB_kp
		self.labels = temp_y