import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import find_contours
from scipy.spatial.distance import cdist
import pandas as pd
from collections import namedtuple
from skimage.morphology import label
import pandas as pd

class Point(namedtuple('Point', 'x y')):
	__slots__ = ()
	@property
	def length(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5
	def __sub__(self, p):
		return Point(self.x - p.x, self.y - p.y)
	def __str__(self):
		return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)

def plot_contours(contours, lw=4, alpha=0.5):
	for n, contour in enumerate(contours):
		plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha)

def vector_norm(p):
	length = p.length
	return Point(p.x / length, p.y / length)

def compute_pairwise_distances(P1, P2, min_length=10):
	euc_dist_matrix = cdist(P1, P2, metric='euclidean')
	indices = []
	for x in range(euc_dist_matrix.shape[0]):
		for y in range(euc_dist_matrix.shape[1]):
			p1 = Point(*P1[x])
			p2 = Point(*P1[y])
			d = euc_dist_matrix[x, y]
			if p1 == p2 or d < min_length:
				continue
			indices.append([p1, p2, d])
	return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)
#
def interpolate(p1, p2, d):
	X = np.linspace(p1.x, p2.x, round(d)).astype(int)
	Y = np.linspace(p1.y, p2.y, round(d)).astype(int)
	XY = np.asarray(list(set(zip(X, Y))))
	return XY
#
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
#
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
#
def find_largest_orthogonal_cross_section(pairwise_distances, img, tolerance=0.1):
	for i, (p1, p2, d1) in enumerate(pairwise_distances):
		# Compute intersections with background pixels
		XY = interpolate(p1, p2, d1)
		intersections = np.sum(img[x, y] == 0 for x, y in XY)
		if intersections/float(len(XY)) < .1:
			V = vector_norm(Point(p2.x - p1.x, p2.y - p1.y))
			# Iterate over remaining line segments
			for j, (q1, q2, d2) in enumerate(pairwise_distances[i:]):
				W = vector_norm(Point(q2.x - q1.x, q2.y - q1.y))
				if abs(np.dot(V, W)) < tolerance:
					XY = interpolate(q1, q2, d2)
					intersections = np.sum(img[x, y] == 0 for x, y in XY)
					if intersections/float(len(XY)) < .1 and intersect(p1,p2,q1,q2):
						return p1, p2, q1, q2
#
def rano(binary_image, tol=0.1, output_file=None, background_image=None, vox_x = 1):
	binary_image2 = binary_image.astype('uint8') * 255
	contours = find_contours(binary_image2, level=1)
	#combine contours
	comb_contours = contours[0]
	for i in range(1,len(contours)):
		comb_contours = np.concatenate((comb_contours,contours[i]))
	comb_contours = comb_contours.astype(int)

	if len(contours) == 0:
		print "No lesion contours > 1 pixel detected."
		return 0.0, 0.0

	# Calculate pairwise distances over all boundaries
	euc_dist_matrix, ordered_diameters = compute_pairwise_distances(comb_contours, comb_contours, min_length=12/np.float(vox_x))

	# Exhaustive search for longest valid line segment and its orthogonal counterpart
	try:
		p1, p2, q1, q2 = find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
		rano_measure = ((p2 - p1).length * (q2 - q1).length) * vox_x * vox_x
	except TypeError:
		print "Error: unable to compute RANO measurement"
		return 0.0, 0.0

	if output_file is not None:
		fig = plt.figure(figsize=(10, 10), frameon=False)
		plt.margins(0,0)
		plt.gca().set_axis_off()
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		if background_image is not None:
			plt.imshow(background_image, cmap='gray')
		else:
			plt.imshow(binary_image, cmap='gray')
		plot_contours(contours, lw=1, alpha=1.)
		D1 = np.asarray([[p1.x, p2.x], [p1.y, p2.y]])
		D2 = np.asarray([[q1.x, q2.x], [q1.y, q2.y]])
		plt.plot(D1[1, :], D1[0, :], lw=2, c='r')
		plt.plot(D2[1, :], D2[0, :], lw=2, c='r')
		plt.text(20, 20, 'RANO: {:.2f}'.format(rano_measure), {'color': 'r', 'fontsize': 20})
		plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0, dpi=100)
		plt.close(fig)
	return (p2 - p1).length, (q2 - q1).length
#
#vox_x = dimensions of voxel in x/y, vox_z = dimensions of voxel in z, pic_dir = directory to save the picture, background = image to show in the background
def get_rano(mask,vox_x,vox_z,pic_dir,background):
	#split connected components
	conn_comp = label(np.round(mask),connectivity=3).astype(np.float)
	all_rano= np.zeros(len(np.unique(conn_comp)))
	j_idx = np.zeros(len(np.unique(conn_comp)))
	for i in range(1,len(np.unique(conn_comp))):
		#make a mask with only the current connected component
		curr_mask = np.zeros(conn_comp.shape)
		idx = np.where(conn_comp==np.unique(conn_comp)[i])
		curr_mask[idx] = 1.0
		curr_mask = curr_mask.astype(np.int)
		curr_maj = np.zeros(curr_mask.shape[2])
		curr_min = np.zeros(curr_mask.shape[2])
		for j in list(np.sum(curr_mask,(0,1)).argsort()[::-1]):
			if np.sum(curr_mask) > 300:
				if np.sum(curr_mask[...,j]>0):
					curr_maj[j], curr_min[j] = rano(np.swapaxes(curr_mask[...,j],0,1), output_file=pic_dir+str(i)+'_'+str(j)+'.png', background_image=np.swapaxes(background[...,j],0,1), vox_x = vox_x)
			if np.sum(curr_maj) > 0:
				break
		curr_maj = curr_maj * vox_x
		curr_min = curr_min * vox_x
		#threshold is twice the slice thickness + gap spacing
		thres = 2*vox_z
		if thres < 10:
			thres = 10
		print(thres)
		curr_maj[np.where(curr_maj<thres)] = 0
		curr_min[np.where(curr_min<thres)] = 0
		curr_rano = np.multiply(curr_maj,curr_min)
		all_rano[i] = curr_rano[np.argmax(curr_rano)]
		j_idx[i] = np.argmax(curr_rano)
	#sum up the top 5 lesions
	idx2 = np.nonzero(all_rano)[0]
	if len(idx2) < 5:
		sum_rano = np.sum(all_rano)
		i_idx = idx2
	else:
		sum_rano = np.sum(all_rano[np.argsort(all_rano)[-5:]])
		i_idx = np.argsort(all_rano)[-5:]
	df2 = pd.DataFrame(np.vstack((i_idx,j_idx[i_idx])).T)
	df2.to_csv(pic_dir+'rano_idx.csv', index=False, header=False, sep=' ')
	return sum_rano