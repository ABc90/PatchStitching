import os
import argparse
import mayavi.mlab as mlab
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import utils.visualization as vis

def rgb_2_scalar_idx(r, g, b):
    return 256**2 *r + 256 * g + b

def normal2rgb(normals):
    '''
    converts the normal vector to RGB color values
    :param normals: nx3 normal vecors
    :return: rgb - Nx3 color value for each normal vector in uint8
    '''
    if normals.shape[1] != 3:
        raise ValueError('normal vector should be n by 3 array')

    normals = np.divide(normals,np.tile(np.expand_dims(np.sqrt(np.sum(np.square(normals), axis=1)),
                                                       axis= 1),[1, 3]))  # make sure normals are normalized

    rgb = 127.5 + 127.5 * normals
    return rgb/255.0


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
    error_o = np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn)))))

    return error_o, error,ang, pgp10_shape, pgp5_shape

def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (trained models)')
    parser.add_argument('--modelname', type=str, default=['stacked_with_dgcnn_experts3_normal_estimation_nopca2_with_atten256-48_64_3',
                                                          'stacked_with_dgcnn_experts3_normal_estimation_nopca2_with_atten512-48_64_2',
                                                          'stacked_with_dgcnn_experts3_normal_estimation_nopca2_with_atten-48_48',
                                                          ], help='model folder')


    parser.add_argument('--indir', type=str, default='./data/pcpnet',
                        help='input folder (point clouds)')
    parser.add_argument('--dataset', type=str, default='testset_temp2.txt', help='shape set file name')
    return parser.parse_args()

opt = parse_arguments()

root = opt.indir
shape_list_filename = opt.dataset
l = len(opt.modelname)

with open(os.path.join(root, shape_list_filename)) as f:
    shape_names = f.readlines()
shape_names = [x.strip() for x in shape_names]
shape_names = list(filter(None, shape_names))
rmss = []
for shape_name in shape_names:
    shape_xyz = os.path.join(opt.indir, shape_name + '.xyz')
    pts = np.loadtxt(shape_xyz).astype('float32')
    pidx_name = os.path.join(opt.indir, shape_name + '.pidx')
    sparse_ids = np.loadtxt(pidx_name).astype('int')
    normal_name = os.path.join(opt.indir, shape_name + '.normals')
    normals = np.loadtxt(normal_name).astype('float32')
    normal_gt_norm = l2_norm(normals)
    normals = np.divide(normals, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))
    angles_error = []
    pgp10_error = []
    pgp5_error = []
    angles_error_o = []
    for i in range(l):#l
        model_outdir = os.path.join(opt.outdir, opt.modelname[i])
        shape = os.path.join(model_outdir, shape_name + '.normals')
        pre_n = np.loadtxt(shape).astype('float32')
        pre_n_norm = l2_norm(pre_n)
        pre_n = np.divide(pre_n, np.tile(np.expand_dims(pre_n_norm, axis=1), [1, 3]))
        rms_o,rms, ang_err, pgp10, pgp5 = error_evalaution_rms(pre_n, normals)
        angles_error.append(rms)
        pgp10_error.append(pgp10)
        pgp5_error.append(pgp5)
        angles_error_o.append(rms_o)
        rmss.append(rms)
        # show models
        if False:
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.8, 0.8, 0.8), figure=str(i))
            mlab.points3d(pts[:, 0], pts[:, 1],
                      pts[:, 2], scale_mode='vector', colormap='Reds', scale_factor=.05)
            mlab.show()

        # show angle errors
        if True:
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.8, 0.8, 0.8), figure=str(i))
            mlab.points3d(pts[:, 0], pts[:, 1],
                          pts[:, 2], ang_err, scale_mode='vector', colormap='Reds', scale_factor=.05)
            mlab.show()


        # show pgp 5,10,...
        if False:
            pgp = np.ones([pts.shape[0]])
            pgp[ang_err <=10]=0.5
            pgp[ang_err <=5] =0
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.8, 0.8, 0.8), figure=str(i))
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.8, 0.8, 0.8), figure=str(i))
            mlab.points3d(pts[:, 0], pts[:, 1],
                          pts[:, 2], pgp, scale_mode='vector', colormap='jet', scale_factor=.05)
            mlab.show()


        # show normals:
        if False:
            colors = normal2rgb(pre_n)
            # N scalars
            scalars = np.zeros((colors.shape[0],))
            for (kp_idx, kp_c) in enumerate(colors):
                scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])

            mlab.figure(bgcolor=(0, 0, 0), fgcolor=(0.8, 0.8, 0.8), figure=str(i))
            mlab.points3d(pts[:, 0], pts[:, 1],
                          pts[:, 2], scalars, scale_mode='vector', scale_factor=.05)
            mlab.show()

    print(shape)
    print('RMS_O per shape: ' + str(angles_error_o))
    print('RMS per shape: ' + str(angles_error))
    print('pgp5 per shape: ' + str(pgp5_error))
    print('pgp10 per shape: ' + str(pgp10_error))

print(str(np.mean(rmss)))






