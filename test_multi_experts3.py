import argparse
import multiprocessing
import os
import sys
import random
import math
import shutil
# import mayavi.mlab as mlab
import numpy as np
import time
import torch
from thop import profile
import copy
import torch.nn.parallel
import torch.utils.data
from scipy import spatial
# from numba import jit
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from data.datasetforpcpnet_dij import PointcloudPatchDatasetKnn, SequentialPointcloudPatchSampler,SequentialShapeRandomPointcloudPatchSampler
from models.DGCNN_With_Shift_and_three_and_atten_experts3 import PatchNet

class combine_normals_multi:
    def __init__(self, pts,gt, weights, kdtree, shape_properties3,inds,pool,points_num,d_bar):
        # self.shape_properties1 = shape_properties1
        # self.shape_properties2 = shape_properties2
        self.shape_properties3 = shape_properties3
        self.pts = pts
        self.gt = gt  # use to debug
        self.weights  = weights
        self.kdtree = kdtree
        self.ind = inds
        self.pool = pool
        self.points_num = points_num
        self.d_bar = d_bar
    def do_select(self,x):
        # local_normals1 = self.shape_properties1[self.ind == x]
        # local_normals2 = self.shape_properties2[self.ind == x]
        a = self.ind == x
        # a = np.where(self.ind == x)
        a = self.ind[x]
        # a = np.searchsorted(self.ind,x,side="left",sorter=None)
        # b =  np.searchsorted(self.ind,x,side="right",sorter=None)
        local_normals3 = self.shape_properties3[a]
        local_weights  = self.weights[a]
        # _, local_pts_id = self.kdtree.query(self.pts[x,:],3) #3,20,180..., defualt= 64,
        if len(local_normals3) > 0:
            # local_pts = self.pts[local_pts_id]
            # local_pts = local_pts-local_pts.mean(0)
            # local_pts_t= np.divide(local_pts, np.tile(np.expand_dims(l2_norm(local_pts), axis=1), [1, 3]))

            # plane_dist = abs(np.matmul(local_normals3,local_pts_t.transpose(1,0))).sum(1)
            # min_p = np.argmin(plane_dist)
            # pred_n = local_normals3[min_p]
            # A = np.exp(-np.power(np.matmul(local_normals3, local_pts_t.transpose(1, 0)),2.)/(np.power(0.1, 2.)))
            # A = abs(np.matmul(local_normals3,local_pts_t.transpose(1,0)))
            # B = np.exp(-np.power(local_pts,2.).sum(1)/(np.power(self.d_bar, 2.))).reshape(-1,1)
            # B = l2_norm(local_pts).reshape(-1,1)
            # B = ((np.power(self.d_bar,2.))*(1-np.exp(-(np.power(l2_norm(local_pts)/self.d_bar, 2.))))).reshape(-1,1)
            # plane_dist =  np.matmul(A,B) #*local_weights.reshape(-1,1)
            # min_p = np.argmin(plane_dist)
            # pred_n = local_normals3[min_p]

            # _, ang, _, _ = error_evalaution_rms(local_normals3, self.gt[x, :]) ## gt
            ### how to chose the best normal?

            # min_id= np.argmin(ang)
            # pred_n = local_normals3[min_id]
            min_w = np.argmax(local_weights)
            pred_n = local_normals3[min_w]

            pred_normals = pred_n
        if len(local_normals3) == 0:
            pred_normals = [0.0,0.0,0.0]

        return pred_normals

    def multiprocess(self):
        items = [ii for ii in range(self.points_num)]
        p = multiprocessing.Pool(self.pool)
        b = p.map(self.do_select, items)
        p.close()
        p.join()
        return b

def combine_normals_test(weights, shape_properties3, inds, points_num, pre_n, gts,kdtree,pts):
    for i in range(points_num):
        # a = inds == i
        # a = np.random.randint(100000,size = 100)
        a = inds[i]
        local_normals3 = shape_properties3[a]
        local_weights = weights[a]

        if len(local_normals3) > 0:
            #==========================================================================
            # _, local_pts_id1 = kdtree.query(pts[i, :], 3)  # 3,20,180..., defualt= 64,
            # _, local_pts_id2 = kdtree.query(pts[i, :], 24)  # 3,20,180..., defualt= 64,
            # _, local_pts_id3 = kdtree.query(pts[i, :], 128)  # 3,20,180..., defualt= 64,
            _, local_pts_id4 = kdtree.query(pts[i, :], 64)  # 3,20,180..., defualt= 64,

            # local_pts1 = pts[local_pts_id1]
            # local_pts1 = local_pts1-local_pts1.mean(0)
            # local_pts1 = np.divide(local_pts1, np.tile(np.expand_dims(l2_norm(local_pts1), axis=1), [1, 3]))

            # local_pts2 = pts[local_pts_id2]
            # local_pts2 = local_pts2-local_pts2.mean(0)
            # local_pts2 = np.divide(local_pts2, np.tile(np.expand_dims(l2_norm(local_pts2), axis=1), [1, 3]))
            #
            # local_pts3 = pts[local_pts_id3]
            # local_pts3 = local_pts3-local_pts3.mean(0)
            # local_pts3 = np.divide(local_pts3, np.tile(np.expand_dims(l2_norm(local_pts3), axis=1), [1, 3]))

            local_pts4 = pts[local_pts_id4]
            local_pts4 = local_pts4-local_pts4.mean(0)
            local_pts4 = np.divide(local_pts4, np.tile(np.expand_dims(l2_norm(local_pts4), axis=1), [1, 3]))

            # local_pts_t= np.divide(local_pts, np.tile(np.expand_dims(l2_norm(local_pts), axis=1), [1, 3]))
            # A = abs(np.matmul(local_normals3, local_pts_t.transpose(1, 0)))
            # B = l2_norm(local_pts).reshape(-1, 1)
            # plane_dist1 = np.median(abs(np.matmul(local_normals3, local_pts1.transpose(1, 0))), axis=1, keepdims=True)
            # plane_dist2 = np.median(abs(np.matmul(local_normals3, local_pts2.transpose(1, 0))), axis=1, keepdims=True)
            # plane_dist3 = np.median(abs(np.matmul(local_normals3, local_pts3.transpose(1, 0))), axis=1, keepdims=True)
            plane_dist4 = np.median(abs(np.matmul(local_normals3, local_pts4.transpose(1, 0))), axis=1, keepdims=True)
            # p = np.concatenate([plane_dist1, plane_dist2, plane_dist3, plane_dist4], axis=-1)
            # p = np.concatenate([plane_dist1, plane_dist4], axis=-1)

            p = np.min(plane_dist4, axis=-1)
            # plane_dist = np.median(abs(np.matmul(local_normals3, local_pts.transpose(1, 0))))
            p = np.exp(-pow(p-p.mean(), 2)/np.var(p))#*local_weights
            min_p = np.argmax(p)
            pre_n[i, :] = local_normals3[min_p]
            ##==========================================================
            # _, ang, _, _ = error_evalaution_rms(local_normals3, gts[i, :])  ## gt
            # min_id = np.argmin(ang)
            # pre_n[i, :] = local_normals3[min_id]
            # ##==========================================================
            # min_w = np.argmax(local_weights)
            # pre_n[i,:] = local_normals3[min_w]
        if len(local_normals3) == 0:
            pre_n[i, :] = [0.0, 0.0, 0.0]
    return pre_n

# @jit(nopython=True)
def combine_normals(weights, shape_properties3, inds, points_num, pre_n): #,w,weights2
    for i in range(points_num):
        # a = inds == i
        # a = np.random.randint(100000,size = 100)
        a = inds[i]
        if len(a)==0:
            pre_n[i,:] = [1.0,0.0,0.0]
        else:
            local_normals3 = shape_properties3[a]
            local_weights = weights[a]
            # local_weights2 = weights2[a]  #######
            # _, local_pts_id = self.kdtree.query(self.pts[x,:],3) #3,20,180..., defualt= 64,
            if len(local_normals3) > 0:
                min_w = np.argmax(local_weights)
                pre_n[i,:] = local_normals3[min_w]
                # w[i,:] = local_weights2[min_w]  #######
            if len(local_normals3) == 0:
                pre_n[i,:] = [1.0,0.0,0.0]
                # w[i,:]=0
    return pre_n  #,w

def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v

def error_evalaution_rms(pred,target):
    """
    Error estimation includes:
    pgp5 (pgp5_shape),pgp10(pgp10_shape) and the RMSE(error)
    of angle error for the unoriented normal

    Changed from Nesti-net by junz
    """
    nn = np.sum(np.multiply(pred, target), axis=1)
    nn[nn > 1] = 1
    nn[nn < -1] = -1
    ang = np.rad2deg(np.arccos(np.abs(nn)))
    error = np.sqrt(np.mean(np.square(ang)))

    # pgp alpha:
    pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))  # portion of good points
    pgp5_shape = sum([j < 5.0 for j in ang]) / float(len(ang))  # portion of good points

    return error,ang, pgp10_shape, pgp5_shape

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default='My evaluation run for pacth normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/media/junz/Volume1/DataSet/nesti_net_data/data/pcpnet', help='input folder (point clouds)')
    # parser.add_argument('--indir', type=str, default='/media/junz/Volume1/DataSet/nesti_net_data/data/pcv-dataset', help='input folder (point clouds)')

    parser.add_argument('--outdir', type=str, default='/media/junz/Volume1/DataSet/StackedNE/results_t2', help='output folder (trained models)')
    parser.add_argument('--models', type=str, default='stacked_with_dgcnn_experts3_normal_estimation_nopca2_with_atten256', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modeldir', type=str, default='/media/junz/Volume1/DataSet/StackedNE/models/stacked_with_dgcnn_experts3_normal_estimation_nopca2_with_atten256', help='model folder')
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    # testset_no_noise,testset_temp,testset_med_noise,testset_vardensity_striped
    parser.add_argument('--dataset', type=str, default='testset_temp.txt', help='shape set file name')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=200, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--points_per_patch', type=float, default=256, nargs='+', help='patch: point number on one patch using knn method')
    parser.add_argument('--visible', type=int, default=False, help='Visualization of the points')
    parser.add_argument('--sampling', type=str, default='sequential_shapes_random_patches', help='sampling strategy, any of:\n'
                                                                     'full: evaluate all points in the dataset\n'
                                                                     'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1152, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)') ##48*64,48*32
    return parser.parse_args()

def eval_patchnormal(opt):

    opt.models = opt.models.split()
    if opt.seed<0:
        opt.seed = random.randint(1, 10000)

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    for model_name in opt.models:
        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        model_filename = os.path.join(opt.modeldir, model_name + opt.modelpostfix)
        param_filename = os.path.join(opt.modeldir, model_name + opt.parmpostfix)

        #load model and training params
        trainopt = copy.deepcopy(torch.load(param_filename))

        if opt.batchSize == 0:
            model_batchSize = trainopt.batchSize
        else:
            model_batchSize = opt.batchSize

        # get indices in targets and predictions corresponding to each output
        pred_dim = 0
        output_pred_ind = []
        target_features = []

        for o in trainopt.outputs:
            if o == 'unoriented_normals' or o == 'oriented_normals':
                if 'normal' not in target_features:
                    target_features.append('normal')
                output_pred_ind.append(pred_dim)
                pred_dim += 3
            elif o == 'max_curvature' or o == 'min_curvature':
                if o not in target_features:
                    target_features.append(o)
                output_pred_ind.append(pred_dim)
                pred_dim += 1
            else:
                raise ValueError('Unknown output: %s' % (o))

        dataset = PointcloudPatchDatasetKnn(
                root=opt.indir,
                shape_list_filename=opt.dataset,
                points_per_patch=trainopt.points_per_patch,
                patch_features=target_features,         # get the gt of each patch for debug and visualization
                point_count_std=trainopt.patch_point_count_std,
                seed=opt.seed,
                identical_epochs=trainopt.identical_epochs,
                use_pca=trainopt.use_pca,
                use_dijkstra=trainopt.use_dijkstra,     # we have not added this function now
                center=trainopt.patch_center,
                point_tuple=trainopt.point_tuple,
                cache_capacity=trainopt.cache_capacity
            )

        # the sample strategy is not good! we hope to uniform include the input shape
        if opt.sampling == 'full':
            datasampler = SequentialPointcloudPatchSampler(dataset)
        elif opt.sampling == 'sequential_shapes_random_patches':
            datasampler = SequentialShapeRandomPointcloudPatchSampler(
                dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                sequential_shapes=True,
                identical_epochs=False)


        dataloader = torch.utils.data.DataLoader(dataset,
                                                sampler=datasampler,
                                                batch_size=model_batchSize,
                                                num_workers=int(opt.workers))

        model = PatchNet(opt=opt,
                         num_points=opt.points_per_patch,
                         use_point_stn=trainopt.use_point_stn,
                         use_point_shift=trainopt.use_point_shift,
                         is_multi_scale=trainopt.is_multi_scale,
                         add_atten = trainopt.add_atten)

        # model = PatchNet(opt = opt,
        #                  num_points = opt.points_per_patch,
        #                  use_point_stn=trainopt.use_point_stn,
        #                  use_point_shift=trainopt.use_point_shift,
        #                  is_multi_scale = trainopt.is_multi_scale)  # need be added by junz


        model.load_state_dict(copy.deepcopy(torch.load(model_filename)))

        model.to(device)
        model.eval()
        ##  params
        # params = model.parameters()
        # input = torch.randn(10,512,3,dtype=torch.float).to(device)
        # r = torch.randn(1,dtype=torch.float).to(device)
        # macs, params = profile(model, inputs=(input,r),)
        # print(" the param of model is : %.2f | %.2f" % (params / (1000 ** 2), macs / (1000 ** 3)))
        params = list(model.parameters())  # 所有参数放在params里
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j  # 每层的参数存入l，这里也可以print 每层的参数
            k = k + l  # 各层参数相加
        print("all params:" + str(k/100000))  # 输出总的参数

        shape_patch_offset = 0
        shape_ind = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

        # append model name to output directory and create directory if necessary
        model_outdir = os.path.join(opt.outdir, model_name)
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)
        shape_patch_offset = 0
        shape_properties1 = torch.zeros(shape_patch_count, opt.points_per_patch, pred_dim, dtype=torch.float, device=device)
        shape_properties2 = torch.zeros(shape_patch_count, opt.points_per_patch, pred_dim, dtype=torch.float, device=device)
        shape_properties3 = torch.zeros(shape_patch_count, opt.points_per_patch, pred_dim, dtype=torch.float, device=device)
        shape_ids = torch.zeros(shape_patch_count, trainopt.points_per_patch, dtype=torch.int32, device=device)
        weights = torch.zeros(shape_patch_count, trainopt.points_per_patch, dtype=torch.float, device=device)
        weights2 = torch.zeros(shape_patch_count, trainopt.points_per_patch, dtype=torch.float, device=device)
        shape_gt = torch.zeros(shape_patch_count, opt.points_per_patch, pred_dim, dtype=torch.float, device=device)

        rms_list = []
        pgp10_list = []
        pgp5_list= []
        total_time = []

        time_start = time.time()
        for batchind, data in batch_enum:
            points = data[0]
            target = data[1]
            normal_gt = data[1]
            data_trans = data[2]
            inds = data[3]
            offsets = data[4]
            scales = data[5]
            points = points.to(device)
            offsets = offsets.to(device)
            scales = scales.to(device)
            data_trans = data_trans.to(device)
            inds = inds.to(device)

            output_loss_weight_diff = (
                        points - points.mean(1).reshape(-1, 1, 3).repeat(1, opt.points_per_patch, 1)).pow(2).sum(2)
            output_loss_weight = output_loss_weight_diff / output_loss_weight_diff.max(1)[0].reshape(-1, 1).repeat(1, opt.points_per_patch)
            output_loss_weight = torch.exp(-output_loss_weight /(0.8))  # we nedd try different sigma??here 0.3, 0.5, 0.7, 0.9
            output_loss_weight.to(device)

            with torch.no_grad():
                r = 0
                # pred1, pred2, pred3, trans = model(points)
                pred1, pred2, pred3, trans, offsets, scales_v, attens = model(points,r=r)
                # pred1, pred2, pred3, trans, offsets, scales_v = model(points)
                # pred1, pred2, pred3, trans, offsets, scales_v = model(points, r=0)

                pred1 = pred1.transpose(1, 2).contiguous()
                pred2 = pred2.transpose(1, 2).contiguous()
                pred3 = pred3.transpose(1, 2).contiguous()

                if trainopt.is_multi_scale == True:
                    scales_v_values = scales_v.cpu().numpy()
                    ids = np.argmax(scales_v_values, axis=1)
                    ids = np.reshape(ids, [-1, 1])
                    labels = torch.LongTensor(ids)
                    scales_v_onehot = torch.zeros(model_batchSize * opt.points_per_patch, 3).scatter_(1, labels, 1)
                    scales_v_onehot = scales_v_onehot.unsqueeze(-1).repeat([1, 1, 3]).reshape([-1, 9])
                    pred_t = torch.cat([pred1,pred2,pred3],dim=2).view(model_batchSize*opt.points_per_patch,9)
                    aa = pred_t[scales_v_onehot == 1.0]
                    aa = aa.reshape(-1,3)
                    pred3 = aa.reshape(model_batchSize,opt.points_per_patch, 3)
                    # output_loss_weight_l = output_loss_weight
                    output_loss_weight = output_loss_weight*torch.max(scales_v, dim=1)[0]
                    output_loss_weight_s = torch.max(scales_v,dim=1)[0]

                print("batchind: " + str(batchind))

            for oi, o in enumerate(trainopt.outputs):
                if o == 'unoriented_normals' or o == 'oriented_normals':
                    o_pred1 = pred1[:, :, output_pred_ind[oi]:output_pred_ind[oi] + 3]
                    o_pred2 = pred2[:, :, output_pred_ind[oi]:output_pred_ind[oi] + 3]
                    o_pred3 = pred3[:, :, output_pred_ind[oi]:output_pred_ind[oi] + 3]

                    if trainopt.use_point_stn:
                        # transform predictions with inverse transform
                        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                        o_pred1[:, :] = torch.bmm(o_pred1, trans.transpose(2, 1))
                        o_pred2[:, :] = torch.bmm(o_pred2, trans.transpose(2, 1))
                        o_pred3[:, :] = torch.bmm(o_pred3, trans.transpose(2, 1))

                    # normalize normals
                    o_pred_len1 = torch.max(o_pred1.new_tensor([sys.float_info.epsilon * 100]),
                                           o_pred1.norm(p=2, dim=2, keepdim=True))
                    o_pred1 = o_pred1 / o_pred_len1

                    o_pred_len2 = torch.max(o_pred2.new_tensor([sys.float_info.epsilon * 100]),
                                           o_pred2.norm(p=2, dim=2, keepdim=True))
                    o_pred2 = o_pred2 / o_pred_len2

                    o_pred_len3 = torch.max(o_pred3.new_tensor([sys.float_info.epsilon * 100]),
                                           o_pred3.norm(p=2, dim=2, keepdim=True))
                    o_pred3 = o_pred3 / o_pred_len3

                    if trainopt.use_pca:
                        # transform predictions with inverse pca rotation (back to world space)
                        # o_pred1_pca = torch.bmm(o_pred1, data_trans.transpose(2, 1))
                        # o_pred2_pca = torch.bmm(o_pred2, data_trans.transpose(2, 1))
                        o_pred3_pca = torch.bmm(o_pred3, data_trans.transpose(2, 1))

                elif o == 'max_curvature' or o == 'min_curvature':
                    o_pred1 = pred1[:, output_pred_ind[oi]:output_pred_ind[oi] + 1]
                    # undo patch size normalization:
                    o_pred1[:, :] = o_pred1 / dataset.patch_radius_absolute[shape_ind][0]
                else:
                    raise ValueError('Unsupported output type: %s' % (o))

                if False: #opt.visible == True: #False:
                    s = np.ones(opt.points_per_patch)

                    points_values = points.clone()

                    if trainopt.use_point_stn == True:
                        points_values = torch.bmm(points_values, trans)
                    points_values_o = points_values
                    if trainopt.use_point_shift ==True:
                        points_values = points_values.transpose(2, 1)
                        points_values = points_values+r*offsets
                        points_values = points_values.transpose(2, 1)

                    points_values = points_values.cpu().numpy()
                    points_values_o = points_values_o.cpu().numpy()
                    points_v = points.cpu().numpy()
                    # scales_values = scales.cpu().numpy()
                    # offsets_values = offsets.cpu().numpy()

                    pred_values3 = o_pred3.cpu().numpy()
                    pred_values2 = o_pred2.cpu().numpy()
                    pred_values1 = o_pred1.cpu().numpy()
                    if trainopt.use_point_shift ==True:
                        offsets_value = offsets.cpu().numpy()

                    o_target_len = torch.max(target.new_tensor([sys.float_info.epsilon * 100]),
                                             target.norm(p=2, dim=2, keepdim=True))
                    o_target = target / o_target_len
                    target_values = o_target.cpu().numpy()

                    for index_b in range(model_batchSize):
                        error1,ang1,_,_= error_evalaution_rms(pred_values1[index_b, :], target_values[index_b, :])
                        error2,ang2,_,_ = error_evalaution_rms(pred_values2[index_b, :], target_values[index_b, :])
                        error3,ang3,_,_ = error_evalaution_rms(pred_values3[index_b, :], target_values[index_b, :])

                        # print("the patch errsor1 is: " + str(error1))
                        # print("the patch errsor2 is: " + str(error2))
                        # print("the patch errsor3 is: " + str(error3))

                        # mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                        #               points_values[index_b, :, 2], ang1/90, scale_mode='vector', colormap='Blues', scale_factor=.05)
                        # mlab.show()
                        # mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                        #               points_values[index_b, :, 2], ang2/90, scale_mode='vector', colormap='Blues', scale_factor=.05)
                        # mlab.show()
                        mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                      points_values[index_b, :, 2], ang3/90, scale_mode='vector', colormap='Blues', scale_factor=.05)
                        mlab.show()

                        mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                      points_values[index_b, :, 2], scales_v_values[index_b, 0,:], scale_mode='vector', colormap='Blues', scale_factor=.05)
                        mlab.show()

                        mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                      points_values[index_b, :, 2], scales_v_values[index_b, 1,:], scale_mode='vector', colormap='Blues', scale_factor=.05)
                        mlab.show()

                        mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                      points_values[index_b, :, 2], scales_v_values[index_b, 2,:], scale_mode='vector', colormap='Blues', scale_factor=.05)
                        mlab.show()

                        if trainopt.add_atten == True:
                            attens_v = attens.cpu().numpy()
                            for j in range(0,attens.shape[1],100):
                                mlab.points3d(points_values[index_b,j, 0], points_values[index_b, j, 1],
                                              points_values[index_b, j, 2], color = (1,1,0), scale_mode='vector', scale_factor=0.05)
                                mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                          points_values[index_b, :, 2], attens_v[index_b, j,:],
                                          scale_mode='vector', colormap='Blues', scale_factor=.05)
                                mlab.show()
                        if trainopt.use_point_shift == True:
                            # output_loss_weight_l_v= output_loss_weight_l.cpu().numpy()

                            offsets_value_scales = l2_norm(offsets_value)
                            # offsets_value_scales = np.multiply(offsets_value_scales, output_loss_weight_l_v)

                            mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                          points_values[index_b, :, 2], offsets_value_scales[index_b,:],
                                          scale_mode='vector', colormap='Blues', scale_factor=.05)
                            mlab.show()

                            mlab.quiver3d(points_values_o[index_b, :, 0], points_values_o[index_b, :, 1],
                                      points_values_o[index_b, :, 2], offsets_value[index_b,0,:],offsets_value[index_b,1,:],offsets_value[index_b,2,:],colormap='Blues')
                            mlab.show()

                        mlab.quiver3d(points_v[index_b, :, 0], points_v[index_b, :, 1],
                                      points_v[index_b, :, 2],
                                      pred_values3[index_b, :, 0], pred_values3[index_b, :, 1],
                                      pred_values3[index_b, :, 2])
                        mlab.quiver3d(points_v[index_b, :, 0], points_v[index_b, :, 1],
                                      points_v[index_b, :, 2],
                                      target_values[index_b, :, 0], target_values[index_b, :, 1],
                                      target_values[index_b, :, 2], colormap='Blues')
                        mlab.show()

                        ## we hope to show the original data:
                        if trainopt.use_pca:
                        # transform predictions with inverse pca rotation (back to world space)
                        #     pts_mean = points_values.mean(0)
                        #     patch_pts = patch_pts - pts_mean
                            points_values= torch.bmm(points, data_trans.transpose(2, 1))

                            points_values =  points_values * torch.unsqueeze(torch.unsqueeze(scales,1),2).repeat(1,1024,3)
                            points_values = points_values+ torch.unsqueeze(offsets,1).repeat(1,1024,1)
                            points_values = points_values.cpu().numpy()
                            mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                                          points_values[index_b, :, 2], ang3/90,
                                          scale_mode='vector', colormap='Blues', scale_factor=.01)

                    if ((batchind+1)*opt.batchSize)%opt.patches_per_shape ==0:
                        mlab.show()

            # select the multi normals on one shape
            batch_offset = 0
            while batch_offset<points.size(0):
                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = points.size(0)-batch_offset

                # append estimated patch properties batch to properties for the current shape
                if trainopt.use_pca:
                    # shape_properties1[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                    #     o_pred1_pca[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                    # shape_properties2[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                    #     o_pred2_pca[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                    shape_properties3[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                        o_pred3_pca[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                else:
                    # shape_properties1[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                    #     o_pred1[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                    # shape_properties2[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                    #     o_pred2[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                    shape_properties3[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                        o_pred3[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                shape_ids[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] = \
                    inds[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                weights[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] = \
                    output_loss_weight[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                weights2[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] = \
                    output_loss_weight_s[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :] #######

                shape_gt[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining,batch_patches_remaining),:] =\
                    normal_gt[batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining<=batch_patches_remaining:


                    # pred_normals = torch.zeros(pts.shape[0], pred_dim, dtype=torch.float, device=device)

                    # we should use a smart method to select a best normal for each point with multi normals
                    # for ii in range(pts.shape[0]):
                    #     local_normals = shape_properties[shape_ids==ii]
                    #     if len(local_normals)>0:
                    #        pred_n = local_normals[0]
                    #        pred_normals[ii,:] = pred_n
                    #     if len(local_normals)==0:
                    #         pass
                    # pred_normals = pred_normals.cpu().numpy()

                    l = shape_properties1.shape[0]*shape_properties1.shape[1]

                    # shape_properties1_v = shape_properties1.view(l, 3)
                    # shape_properties2_v = shape_properties2.view(l, 3)
                    shape_properties3_v = shape_properties3.view(l, 3)
                    weights_v = weights.view(l)
                    # weights2_v = weights2.view(l)
                    shape_ids_v = shape_ids.view(l)

                    # shape_properties1_v = shape_properties1_v.cpu().numpy()
                    # shape_properties2_v = shape_properties2_v.cpu().numpy()
                    shape_properties3_v = shape_properties3_v.cpu().numpy()
                    shape_ids_v = shape_ids_v.cpu().numpy()
                    weights_v = weights_v.cpu().numpy()
                    # weights_s = weights2_v.cpu().numpy()

                    # kdtree = spatial.cKDTree(pts, 10)
                    # select1 = combine_normals_multi(pts, kdtree, shape_properties1_v,shape_ids_v,pool=64,points_num=pts.shape[0])
                    # data1 = select1.multiprocess()
                    # pred_normals1 =np.array([[data1[ii][0],data1[ii][1],data1[ii][2]] for ii in range(pts.shape[0])])
                    #
                    # select2 = combine_normals_multi(pts, kdtree, shape_properties2_v,shape_ids_v,pool=64,points_num=pts.shape[0])
                    # data2 = select2.multiprocess()
                    # pred_normals2 =np.array([[data2[ii][0],data2[ii][1],data2[ii][2]] for ii in range(pts.shape[0])])



                    # a = np.argsort(shape_ids_v)
                    # shape_ids_v = shape_ids_v[a]
                    # shape_properties3_v = shape_properties3_v[a,:]

                    # l, _ = kdtree.query(pts, 2)
                    # d_bar = l[:,1].mean()

                    ####
                    cols = np.arange(shape_ids_v.size)
                    # M = csc_matrix((cols,(shape_ids_v.ravel(),cols)), ## csr_matrix
                    #                shape=(shape_ids_v.max()+1,shape_ids_v.size))
                    M = csc_matrix((cols,(shape_ids_v.ravel(),cols)), ## csr_matrix
                                   shape=(100000,shape_ids_v.size))
                    a = [np.unravel_index(row.data, shape_ids_v.shape) for row in M]
                    # from numba.typed import List
                    # typed_a = List()
                    # a = [typed_a.append(np.unravel_index(row.data, shape_ids_v.shape)) for row in M]

                    name = dataset.shape_names[shape_ind]
                    shape_name = os.path.join(opt.indir, name + '.xyz')
                    pts = np.loadtxt(shape_name).astype('float32')
                    pre_n = np.zeros([pts.shape[0], 3])
                    # w = np.zeros([pts.shape[0],1])
                    pred_normals3 = combine_normals(weights_v, shape_properties3_v, a, pts.shape[0], pre_n)#, w, weights_s)




                    # kdtree = spatial.cKDTree(pts, 10)
                    # pred_normals3 = combine_normals_test(weights_v, shape_properties3_v, a, pts.shape[0],pre_n,normals, kdtree, pts)

                    ###
                    # select3= combine_normals_multi(pts,normals,weights_v, kdtree, shape_properties3_v,a,pool=64,points_num=pts.shape[0],d_bar=d_bar)
                    # data3 = select3.multiprocess()
                    # pred_normals3 =np.array([[data3[ii][0],data3[ii][1],data3[ii][2]] for ii in range(pts.shape[0])])


                    time_end = time.time()
                    time_delta = time_end-time_start

                    name = dataset.shape_names[shape_ind]
                    shape_name = os.path.join(opt.indir, name + '.xyz')
                    pts = np.loadtxt(shape_name).astype('float32')
                    # pidx_name = os.path.join(opt.indir, name + '.pidx')
                    # sparse_ids = np.loadtxt(pidx_name).astype('int')
                    normal_name = os.path.join(opt.indir, name + '.normals')
                    normals = np.loadtxt(normal_name).astype('float32')

                    normal_gt_norm = l2_norm(normals)
                    normals = np.divide(normals, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))
                    # SAVE
                    np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind] + '.normals'),
                               pred_normals3)
                    # np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind] + '.weights'),
                    #            w)

                    # normal_results_norm = l2_norm(pred_normals)
                    # pred_normals = np.divide(pred_normals,
                    #                             np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))

                    # rms_1,ang_err_1,pgp10_1,pgp5_1 = error_evalaution_rms(pred_normals1, normals)
                    # rms_2,ang_err_2,pgp10_2,pgp5_2 = error_evalaution_rms(pred_normals2, normals)
                    pred_normals3_s = pred_normals3#[sparse_ids,:]
                    normals_s = normals#[sparse_ids,:]
                    rms_3_s,ang_err_3_s,pgp10_3_s,pgp5_3_s = error_evalaution_rms(pred_normals3_s, normals_s)
                    rms_3,ang_err_3,pgp10_3,pgp5_3 = error_evalaution_rms(pred_normals3, normals)

                    # err_c = np.concatenate([np.expand_dims(ang_err_1, 1), np.expand_dims(ang_err_2, 1), np.expand_dims(ang_err_3, 1)], 1)
                    # ang_err = np.min(np.concatenate([np.expand_dims(ang_err_1,1),np.expand_dims(ang_err_2,1),np.expand_dims(ang_err_3,1)],1),1)
                    # rms = np.sqrt(np.mean(np.square(ang_err)))
                    ang_err = ang_err_3
                    rms = rms_3
                    pgp10 = pgp10_3
                    pgp5 = pgp5_3

                    total_time.append(time_delta)
                    rms_list.append(rms_3_s)
                    pgp10_list.append(pgp10_3_s)
                    pgp5_list.append(pgp5_3_s)

                    print('RMS per shape: ' + str(rms_3_s))
                    print('PGP10 per shape: ' + str(pgp10_3_s))
                    print('PGP5 per shape: ' + str(pgp5_3_s))
                    print("time: " + str(time_delta/100)+'ms/p')

                    if opt.visible == True:
                        # ang_err[ang_err > 10] = 90
                        mlab.points3d(pts[:, 0], pts[:, 1],
                                      pts[:, 2],ang_err ,
                                      scale_mode='vector', colormap='jet', scale_factor=0.02) # Blues,jet
                        mlab.show()


                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif opt.sampling == 'sequential_shapes_random_patches':
                            # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                        shape_properties1 = shape_properties1.new_zeros(shape_patch_count, opt.points_per_patch, pred_dim)
                        shape_properties2 = shape_properties2.new_zeros(shape_patch_count, opt.points_per_patch, pred_dim)
                        shape_properties3 = shape_properties3.new_zeros(shape_patch_count, opt.points_per_patch, pred_dim)

                        shape_ids = shape_ids.new_zeros(shape_patch_count, trainopt.points_per_patch)
                        time_start = time.time()

        avg_rms = np.mean(rms_list)
        avg_pgp10 = np.mean(pgp10_list)
        avg_pgp5 = np.mean(pgp5_list)
        ave_time = np.mean(total_time)
        print('RMS per shape: ' + str(rms_list))
        print('RMS not oriented (shape average): ' + str(avg_rms))
        print('PGP10 per shape: ' + str(pgp10_list))
        print('PGP5 per shape: ' + str(pgp5_list))
        print('PGP10 average: ' + str(avg_pgp10))
        print('PGP5 average: ' + str(avg_pgp5))
        print('PGP5 average: ' + str(ave_time/100)+'ms/p')



if __name__ == '__main__':
    eval_opt = parse_arguments()
    eval_patchnormal(eval_opt)