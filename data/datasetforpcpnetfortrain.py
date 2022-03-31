from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial


def graipher(pts, K, init_points_id=[]):
    """
    sample some farthest points on the 3d points
    pts: the points
    K: the sampling points
    nit_points: given the first point
    output: K sampling points
    """
    farthest_pts = np.zeros((K, 3))
    farthest_ind = np.arange(K)
    if not any(init_points_id):
        farthest_ind[0] = np.random.randint(len(pts))
        farthest_pts[0] = pts[farthest_ind[0]]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(K):
            farthest_ind[i] = np.argmax(distances)
            farthest_pts[i] = pts[farthest_ind[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    else:
        l = len(init_points_id)
        farthest_ind[0:l] = init_points_id
        farthest_pts[0:l] = pts[init_points_id]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(l):
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
        for i in range(l, K):
            farthest_ind[i] = np.argmax(distances)
            farthest_pts[i] = pts[farthest_ind[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, farthest_ind

def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename, curv_filename, pidx_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count

class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count



class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDatasetKnn(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shape_list_filename, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=True,
                 use_dijkstra=True, center='point',
                 point_tuple=1, cache_capacity=1,
                 local_disturb = False,
                 point_count_std=0.0, sparse_patches=False):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed
        self.local_disturb = local_disturb # False
        self.include_normals = False
        self.include_curvatures = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDatasetKnn.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                curvatures = np.loadtxt(curv_filename).astype('float32')
                np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None
                # centers, pidx = graipher(pts, 8000, []) #train use large num 8000
            shape = self.shape_cache.get(shape_ind)
            # shape.pidx = pidx

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]
            # centers, center_point_ind = graipher(shape.pts, 2000, [])

        # get neighboring points (using knn methdo to obtain fixed number of points)
        patch_pts = torch.zeros(self.points_per_patch, 3, dtype=torch.float)
        offset = torch.zeros(3, dtype=torch.float)
        scale = torch.ones(1, dtype=torch.float)

        dist, patch_point_inds = shape.kdtree.query(shape.pts[center_point_ind, :], self.points_per_patch)
        patch_point_inds = np.array(patch_point_inds)
        # optionally always pick the same points for a given patch index (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed((self.seed + index) % (2**32))

        # convert points to torch tensors
        patch_pts = torch.from_numpy(shape.pts[patch_point_inds, :])

        # center patch (central point at origin - but avoid changing padded zeros)
        if self.center == 'mean':
            patch_pts = patch_pts - patch_pts.mean(0)
            offset = patch_pts.mean(0)
        elif self.center == 'point':
            patch_pts = patch_pts - torch.from_numpy(shape.pts[center_point_ind, :])
            offset = torch.from_numpy(shape.pts[center_point_ind, :])

        elif self.center == 'none':
            pass # no centering
        else:
            raise ValueError('Unknown patch centering option: %s' % (self.center))

        # normalize size of patch (scale with 1 / patch radius)
        # patch_pts = patch_pts[start:end, :] / rad
        patch_pts = patch_pts/dist[len(dist)-1]
        scale = dist[len(dist)-1]
        # patch_pts = patch_pts



        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[patch_point_inds, :])

        if self.local_disturb == True:
            a = np.random.rand()
            if a >0.5:
                # patch_pts_t = patch_pts.clone()
                shifts = np.random.uniform(-0.05, 0.05, (patch_normal.shape[0], 3))
                shifts = torch.FloatTensor(shifts)
                l = torch.sum(shifts*patch_normal,dim=1)/patch_normal.norm(2,1)
                shift_vector = shifts - l.view(-1,1)* patch_normal
                patch_pts = patch_pts+shift_vector
                # patch_pts = patch_pts_t
            # for k in range(patch_normal.shape[0]):
            #     shift_l = np.random.uniform(0, 0.05)
            #
            #     angles = np.clip(np.pi * np.random.randn(3), -np.pi, np.pi)
            #     Rx = np.array([[1, 0, 0],
            #                [0, np.cos(angles[0]), -np.sin(angles[0])],
            #                [0, np.sin(angles[0]), np.cos(angles[0])]])
            #     Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
            #                [0, 1, 0],
            #                [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            #     Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
            #                [np.sin(angles[2]), np.cos(angles[2]), 0],
            #                [0, 0, 1]])
            #     R = np.dot(Rz, np.dot(Ry, Rx))
            #     rot_t = torch.FloatTensor(np.dot([0,0,1],R))
            #     shift_vector = rot_t - (torch.dot(rot_t,patch_normal[k, ...])/patch_normal[k, ...].norm(2))*patch_normal[k, ...]
            #     shift_vector = shift_vector/torch.max(shift_vector.new_tensor(sys.float_info.epsilon * 100),shift_vector.norm(2))
            #     patch_pts_t[k,...] = patch_pts_t[k,...]+ shift_vector*shift_l
            # detal = patch_pts-patch_pts_t
            # print(detal.mul(patch_normal))


        if self.include_curvatures:
            patch_curv = torch.from_numpy(shape.curv[patch_point_inds, :])
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            patch_curv = patch_curv

        if self.use_pca:

            # compute pca of points in the patch:
            # center the patch around the mean:
            pts_mean = patch_pts.mean(0)
            patch_pts = patch_pts - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts))
            patch_pts = torch.mm(patch_pts, trans)

            cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
            cp_new = torch.matmul(cp_new, trans)

            # re-center on original center point
            patch_pts = patch_pts - cp_new

            if self.include_normals:
                patch_normal = torch.matmul(patch_normal, trans)

        else:
            trans = torch.eye(3).float()

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal,)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        return (patch_pts,) + patch_feats + (trans,)+ (patch_point_inds,) +(offset, ) + (scale,)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None

        return load_shape(point_filename, normals_filename, curv_filename, pidx_filename)
