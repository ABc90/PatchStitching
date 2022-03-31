import argparse
import os
import sys
import random
import math
import shutil
import time
# import mayavi.mlab as mlab
import numpy as np
import torch
import copy
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch
import utils.pcpnet_utils as pcpnet_utils
from data.datasetforpcpnetfortrain import PointcloudPatchDatasetKnn, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from models.DGCNN_With_Shift_and_three_and_atten_experts3 import PatchNet

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    #stacked_with_dgcnn_based_normal_estimation
    parser.add_argument('--name', type=str, default='stacked_with_dgcnn_experts3_normal_estimation_nopca2_without_atten1024', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for patch normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/media/junz/Volume1/DataSet/nesti_net_data/data/pcpnet', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='/media/junz/Volume1/DataSet/StackedNE/models/stacked_with_dgcnn_experts3_normal_estimation_nopca2_without_atten1024', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='/media/junz/Volume1/DataSet/StackedNE/', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name') #trainingset_whitenoise,testset_temp,trainingset_vardensity_whitenoise
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--visible', type=int, default=False, help='Visualization of the points')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=1200, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size') # step1: 24
    parser.add_argument('--points_per_patch', type=float, default=1024, nargs='+', help='patch: point number on one patch using knn method') #512
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=200, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')

    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate') # setep1: 0.1; step2: 0.01
    parser.add_argument('--weight_decay',type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr_clip',type=float, default=0.0003, help='learning clip')
    parser.add_argument('--lr_decay',type=float, default=0.5, help='learning decay')
    parser.add_argument('--decay_step',type=int, default=21, help='decay step')
    parser.add_argument('--bn_momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--bnm_clip',type=float, default=0.01, help='bnm clip')
    parser.add_argument('--bn_decay',type=float, default=0.5, help='bnm decay')

    parser.add_argument('--local_disturb', type=int, default=False, help='sample local ponts set use local disturb')
    parser.add_argument('--use_dijkstra', type=int, default=True, help='sample local ponts set use Dijkstra algotithm')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--add_atten', type=int, default=False, help='use point spatial transformer')
    parser.add_argument('--use_point_shift', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--is_multi_scale', type=int, default=True, help='use experts choose')
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'],
                        help='outputs of the network, a list with elements of:\n'
                             'unoriented_normals: unoriented (flip-invariant) point normals\n'
                             'oriented_normals: oriented point normals\n'
                             'max_curvature: maximum curvature\n'
                             'min_curvature: mininum curvature')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--patches_per_shape', type=int, default=1200, help='number of patches sampled from each shape in an epoch') #1200



    return parser.parse_args()

def train_pcpnet(opt):
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    # if os.path.exists(log_dirname) or os.path.exists(model_filename):
    #     response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
    #     if response == 'y':
    #         if os.path.exists(log_dirname):
    #             shutil.rmtree(os.path.join(opt.logdir, opt.name))
    #     else:
    #         sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

        if pred_dim <= 0:
            raise ValueError('Prediction is empty for the given outputs.')

        model = PatchNet(opt = opt,
                         num_points = opt.points_per_patch,
                         use_point_stn=opt.use_point_stn,
                         use_point_shift=opt.use_point_shift,
                         is_multi_scale = opt.is_multi_scale,
                         add_atten = opt.add_atten)  # need be added by junz

        if opt.refine != '':
            model.load_state_dict(copy.deepcopy(torch.load(opt.refine)))

        if opt.seed < 0:
            opt.seed = random.randint(1, 10000)

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        # the function need be modified by junz
        train_dataset = PointcloudPatchDatasetKnn(
            root=opt.indir,
            shape_list_filename=train_opt.trainset,
            points_per_patch=opt.points_per_patch,
            patch_features=target_features,
            point_count_std=opt.patch_point_count_std,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs,
            use_pca=opt.use_pca,
            use_dijkstra = opt.use_dijkstra,  # we have not added this function now
            center=opt.patch_center,
            point_tuple=opt.point_tuple,
            cache_capacity=opt.cache_capacity,
            local_disturb = opt.local_disturb

        )

        if opt.training_order == 'random':
            train_datasampler = RandomPointcloudPatchSampler(
                train_dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                identical_epochs=opt.identical_epochs)
        elif opt.training_order == 'random_shape_consecutive':
            train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
                train_dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                identical_epochs=opt.identical_epochs)
        else:
            raise ValueError('Unknown training order: %s' % (opt.training_order))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_datasampler,
            batch_size=opt.batchSize,
            num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names

    print('training set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass


    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))

    for k, v in model.named_parameters():
        # print('{}: {}'.format(k, v.requires_grad))
        if 'shift_net'  in k:  #
        # if 'shift_net' not in k:  #
            v.requires_grad = False

    for k, v in model.named_parameters():
            print('{}: {}'.format(k, v.requires_grad))


    params = list(model.parameters())  # 所有参数放在params里
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j  # 每层的参数存入l，这里也可以print 每层的参数
        k = k + l  # 各层参数相加
    print("all params:" + str(k/100000))  # 输出总的参数



    # optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, momentum=opt.bn_momentum,weight_decay=opt.weight_decay) # pcpnet: weight_decay=0
    optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=opt.base_lr, momentum=opt.bn_momentum,weight_decay=opt.weight_decay) # pcpnet: weight_decay=0
    # lr_scheduler = lr_sched.MultiStepLR(optimizer, milestones=[], gamma=0.1) # m
    lr_scheduler = lr_sched.StepLR(optimizer, step_size=50, gamma=0.5) # default = 100
    # lr_scheduler = lr_sched.MultiStepLR(optimizer, milestones=[200, 500, 700], gamma=0.1)




    # optimizer = optim.Adam(
    #     model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay)
    # lr_lbmd = lambda e: max(opt.lr_decay**(e // opt.decay_step), opt.lr_clip / opt.base_lr)
    # lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    # bnm_lmbd = lambda e: max(opt.bn_momentum * opt.bn_decay**(e // opt.decay_step), opt.bnm_clip)
    # bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    model.to(device)

    train_num_batch = len(train_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)


    for epoch in range(opt.nepoch):
        time_start = time.time()
        train_enum = enumerate(train_dataloader, 0)

        for train_batchind, data in train_enum:
            # if lr_scheduler is not None:
            #     lr_scheduler.step(epoch)
            # if bnm_scheduler is not None:
            #     bnm_scheduler.step(epoch-1)

            points = data[0]
            target = data[1:-4]
            normal_gt = data[1]
            points_xyz = points
            # points = points.transpose(2, 1)
            # normal_gt = normal_gt.transpose(2, 1)

            points = points.to(device)

            output_loss_weight_diff = (
                        points - points.mean(1).reshape(-1, 1, 3).repeat(1, opt.points_per_patch, 1)).pow(2).sum(2)
            output_loss_weight = output_loss_weight_diff / output_loss_weight_diff.max(1)[0].reshape(-1, 1).repeat(1,
                                                                                                                   opt.points_per_patch)
            output_loss_weight = torch.exp(-output_loss_weight / 0.8)
            output_loss_weight_value = output_loss_weight.cpu()

            if opt.visible==True:
                print("the size of train data is : " + str(points.shape))
                print("the size of train normal_gt is : " + str(normal_gt.shape))
                s = np.ones(opt.points_per_patch)
                for i in range(opt.batchSize):
                # mlab.points3d(points_xyz[i, :, 0], points_xyz[i, :, 1], points_xyz[i, :, 2], 0.2*s, scale_factor=.25)
                    mlab.points3d(points_xyz[i, :, 0], points_xyz[i, :, 1], points_xyz[i, :, 2], output_loss_weight_value[i, :],scale_mode = 'vector', colormap = 'Blues', scale_factor = .05)
                    mlab.show()

            target = tuple(t.to(device) for t in target)

            # zero gradients
            optimizer.zero_grad()
            # if epoch<=100000:
            #     r=0
            # else:
            #     r=1

            # forward pass
            if False:
                # pred1_pre,pred2_pre,pred3_pre, trans_pre, _, scales_pre = model(points,r=0)
                pred1_pre,pred2_pre,pred3_pre, trans_pre, _, scales_pre, attens_pre = model(points, r=0)

                pred1_pre = pred1_pre.transpose(1, 2).contiguous()
                pred2_pre = pred2_pre.transpose(1, 2).contiguous()
                pred3_pre = pred3_pre.transpose(1, 2).contiguous()

                if opt.is_multi_scale == True:
                    output_loss_weight1_pre = torch.mul(scales_pre[:, 0, :], output_loss_weight)
                    output_loss_weight2_pre = torch.mul(scales_pre[:, 1, :], output_loss_weight)
                    output_loss_weight3_pre = torch.mul(scales_pre[:, 2, :], output_loss_weight)
                elif opt.is_multi_scale == False & (epoch <= 500):
                    output_loss_weight1_pre = 0.33 * output_loss_weight
                    output_loss_weight2_pre = 0.33 * output_loss_weight
                    output_loss_weight3_pre = 0.33 * output_loss_weight
                elif opt.is_multi_scale == False & (epoch > 500):
                    output_loss_weight1_pre = 0.01 * output_loss_weight
                    output_loss_weight2_pre = 0.01 * output_loss_weight
                    output_loss_weight3_pre = 0.98 * output_loss_weight

                loss1_p, loss_wise1_p = compute_loss(
                    pred=pred1_pre, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight1_pre,
                    patch_rot=trans_pre if opt.use_point_stn else None,
                    normal_loss=opt.normal_loss)

                loss2_p, loss_wise2_p = compute_loss(
                    pred=pred2_pre, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight2_pre,
                    patch_rot=trans_pre if opt.use_point_stn else None,
                    normal_loss=opt.normal_loss)

                loss3_p, loss_wise3_p = compute_loss(
                    pred=pred3_pre, target=target,
                    outputs=opt.outputs,
                    output_pred_ind=output_pred_ind,
                    output_target_ind=output_target_ind,
                    output_loss_weight=output_loss_weight3_pre,
                    patch_rot=trans_pre if opt.use_point_stn else None,
                    normal_loss=opt.normal_loss)
                loss_wise_pre = torch.min(torch.cat([loss_wise1_p.reshape(opt.batchSize,opt.points_per_patch,1),
                                                    loss_wise2_p.reshape(opt.batchSize,opt.points_per_patch,1),
                                                    loss_wise3_p.reshape(opt.batchSize,opt.points_per_patch,1)],-1),-1,keepdim=True)[0]

                offsets_weights_t = torch.max(loss_wise_pre,1,keepdim=True)[0]
                offsets_weights = loss_wise_pre/offsets_weights_t
                offsets_weights = torch.exp(-offsets_weights.view(opt.batchSize,opt.points_per_patch)/0.1)
                # offsets_weights = 500*offsets_weights/offsets_weights.sum(dim=1,keepdim=True)
                offsets_weights = offsets_weights/offsets_weights.max(dim=1,keepdim=True)[0]
                offsets_weights = offsets_weights+0.8*(1-output_loss_weight)
            # offsets_weights = torch.mul(offsets_weights)
            # offsets_weights = torch.clamp(offsets_weights,min=0.00000001)
            if False:
                points_values = points.cpu().numpy()
                offsets_weights_v = offsets_weights.cpu().numpy()
                for index_b in range(opt.batchSize):
                    mlab.points3d(points_values[index_b, :, 0], points_values[index_b, :, 1],
                              points_values[index_b, :, 2],offsets_weights_v[index_b], scale_mode='vector', colormap='Blues',
                              scale_factor=.05)
                    mlab.show()

            ########

            pred1,pred2,pred3, trans, offsets, scales,attens = model(points,r=0)
            # pred1,pred2,pred3, trans, offsets, scales = model(points)

            pred1 = pred1.transpose(1, 2).contiguous()
            pred2 = pred2.transpose(1, 2).contiguous()
            pred3 = pred3.transpose(1, 2).contiguous()

            if opt.is_multi_scale == True:
                output_loss_weight1 = torch.mul(scales[:,0,:],output_loss_weight)
                output_loss_weight2 = torch.mul(scales[:,1,:],output_loss_weight)
                output_loss_weight3 = torch.mul(scales[:,2,:],output_loss_weight)
            elif opt.is_multi_scale == False & (epoch<=500):
                output_loss_weight1 = 0.33*output_loss_weight
                output_loss_weight2 = 0.33*output_loss_weight
                output_loss_weight3 = 0.33* output_loss_weight
            elif opt.is_multi_scale == False & (epoch>500):
                output_loss_weight1 = 0.01 *output_loss_weight
                output_loss_weight2 = 0.01*output_loss_weight
                output_loss_weight3 = 0.98* output_loss_weight

            loss1,loss_wise1 = compute_loss(
                pred=pred1, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight1,
                patch_rot=trans if opt.use_point_stn else None,
                normal_loss=opt.normal_loss)

            loss2,loss_wise2 = compute_loss(
                pred=pred2, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight2,
                patch_rot=trans if opt.use_point_stn else None,
                normal_loss=opt.normal_loss)

            loss3,loss_wise3 = compute_loss(
                pred=pred3, target=target,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=output_loss_weight3,
                patch_rot=trans if opt.use_point_stn else None,
                normal_loss=opt.normal_loss)


            if opt.use_point_shift == True:
                #step2:
                # loss_shift = shift_loss(offsets, output_loss_weight=offsets_weights)

                loss_shift = shift_loss(offsets, output_loss_weight=output_loss_weight)
                loss_vertical = vertical_offset_loss(offsets = offsets,
                                                     target=target,
                                                     patch_rot=trans if opt.use_point_stn else None,
                                                     output_target_ind= output_target_ind,
                                                     outputs = opt.outputs,
                                                     output_loss_weight = output_loss_weight)
                if epoch<=10000:
                    loss = loss1 + loss2 + loss3
                    # loss = loss1 + loss2 + loss3 + loss_shift #+ 0.000001*loss_vertical
                else:
                    # loss = loss1 + loss2 + loss3 +  0.01*loss_shift + 0.0001*loss_vertical  #0.01 +0.1
                    loss = loss1 + loss2 + loss3 +  0.1*loss_shift + 0.1*loss_vertical  #0.01 +0.1

            if opt.use_point_shift == False:
                loss = loss1 + loss2 + loss3

            if opt.visible==True:
                pred_out3 = pred3.cpu().detach()
                pred_out1 = pred1.cpu().detach()
                pred_out3 = pred3.cpu().detach()

                # loss = 0.33*loss1+0.33*loss2+0.33*loss3
                loss_wise = loss_wise1 + loss_wise2 + loss_wise3
                loss_wise = loss_wise.cpu().detach()

                s = np.ones(opt.points_per_patch)
                for i in range(0,opt.batchSize,10):
                    mlab.points3d(points_xyz[i, :, 0], points_xyz[i, :, 1], points_xyz[i, :, 2], loss_wise[i,:], scale_mode = 'vector',colormap='Blues',scale_factor=.05)
                    # mlab.quiver3d(points_xyz[i,:,0],points_xyz[i,:,1],points_xyz[i,:,2], pred_out1[i,:,0],pred_out1[i,:,1],pred_out1[i,:,2])
                    mlab.show()

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind + 1) / train_num_batch

            # print info and update log file
            if opt.use_point_shift == False:
                print('[%s %d: %d/%d] %s loss1: %f, loss2: %f, loss3: %f' % (
            opt.name, epoch, train_batchind, train_num_batch - 1, green('train'), loss1.item(),loss2.item(),loss3.item()))
            if opt.use_point_shift == True:
                print('[%s %d: %d/%d] %s loss1: %f, loss2: %f, loss3: %f, loss_shift: %f, loss_vertical: %f' % (
                    opt.name, epoch, train_batchind, train_num_batch - 1, green('train'), loss1.item(), loss2.item(),
                    loss3.item(), loss_shift.item(),loss_vertical.item()))


            train_writer.add_scalar('loss', loss.item(),
                                    (epoch + train_fraction_done) * train_num_batch * opt.batchSize)

            lr_scheduler.step(epoch)

        ## cost of one epoch
        time_end = time.time()
        time_delta = time_end-time_start
        print("Cost time of one epoch is "+ str(time_delta))
        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            torch.save(model.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(model.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch+700))) ###


def shift_loss(offsets, output_loss_weight):
    # s = 0.05*torch.ones(offsets.size(0),offsets.size(2), dtype=torch.float)
    # z = torch.zeros(offsets.size(0),offsets.size(2), dtype=torch.float)
    # s = s.to(device)
    # z = z.to(device)
    loss_wise = torch.clamp(offsets.norm(2,1)-0.0001, min=0.00000001) #*(1-output_loss_weight+0.000001)
    loss_temp = torch.mul(loss_wise, output_loss_weight)
    loss = loss_temp.mean()
    return loss

def vertical_offset_loss(offsets, target,patch_rot,output_target_ind,outputs,output_loss_weight):
    loss = 0
    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_target = target[output_target_ind[oi]]

            offsets = offsets.transpose(1, 2).contiguous()
            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                offsets = torch.bmm(offsets, patch_rot.transpose(2, 1))

            cos_v = torch.mul(offsets, o_target).sum(2) / torch.clamp(offsets.norm(2, 2) * o_target.norm(2, 2),min=0.000001)
            loss_temp = torch.mul(torch.abs(cos_v), output_loss_weight)

            loss += loss_temp.mean()
    return loss


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind, output_loss_weight, patch_rot, normal_loss):

    loss = 0

    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_pred = pred[..., output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_target = target[output_target_ind[oi]]

            if patch_rot is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred = torch.bmm(o_pred, patch_rot.transpose(2, 1))

            if o == 'unoriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss_wise = torch.min((o_pred-o_target).pow(2).sum(2), (o_pred+o_target).pow(2).sum(2))
                    loss_temp= torch.mul(loss_wise, output_loss_weight)
                    loss += loss_temp.mean()
                # elif normal_loss == 'ms_oneminuscos':
                #     loss += (1-torch.abs(pcpnet_utils.cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight
                elif normal_loss == 'ms_oneminuscos':
                    batch_size = o_pred.shape[0]
                    n_points = o_pred.shape[1]
                    cos_ang = pcpnet_utils.cos_angle(o_pred.view(-1, 3), o_target.view(-1, 3)).view(batch_size, n_points)
                    loss_wise = (1 - torch.abs(cos_ang)).pow(2)
                    loss_temp= torch.mul(loss_wise, output_loss_weight)
                    loss += loss_temp.mean()
                elif normal_loss == 'sin':
                    loss = 0.25 * torch.mean(output_loss_weight * torch.norm(torch.cross(o_pred.view(-1, 3),
                                                                         o_target.view(-1, 3),dim=-1).view(batch_size, -1, 3), p=2, dim=2))
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred-o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-pcpnet_utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return loss,loss_wise

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)



