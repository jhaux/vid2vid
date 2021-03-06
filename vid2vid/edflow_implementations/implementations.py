from vid2vid.models.models import create_model
from vid2vid.options.train_options import _TrainOptions as TrainOptions
from vid2vid.options.test_options import _TestOptions as TestOptions

from edflow.iterators.model_iterator import PyHookedModelIterator
from edflow.hooks.hook import Hook
from edflow.hooks.util_hooks import IntervalHook
from edflow.hooks.pytorch_hooks import PyCheckpointHook, PyLoggingHook
from edflow.hooks.evaluation_hooks import WaitForCheckpointHook, \
                                          RestorePytorchModelHook, \
                                          MetricTuple, \
                                          MetricHook, \
                                          KeepBestCheckpoints
from edflow.custom_logging import get_logger, init_project
from edflow.project_manager import ProjectManager
from edflow.iterators.batches import plot_batch, save_image
from edflow.util import retrieve, walk
from edflow.metrics.image_metrics import ssim_metric, l2_metric

from triplet_reid.edflow_implementations.implementations import reIdMetricFn

from hbu_journal.keypointutils import keypoint_distance_metric as kp_metric

from hbu_journal.implementations.eval_output import TransferHook

import numpy as np
import os

import torch
from torch.autograd import Variable


logger = get_logger('vid2vid')
P = ProjectManager()


def reshape(tensors):
    '''Joins temporal and batch dimension.'''
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    if tensors is None:
        return None
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)


def get_skipped_frames(B_all, B, t_scales, tD):
    ''' get temporally subsampled frames for real/fake sequences'''
    B_all = torch.cat([B_all.detach(), B], dim=1) if B_all is not None else B
    B_skipped = [None] * t_scales
    for s in range(t_scales):
        # number of skipped frames between neighboring frames (e.g. 1, 3, 9,
        # ...)
        tDs = tD ** s
        # number of frames the final triplet frames span before skipping (e.g.,
        # 2, 6, 18, ...)
        span = tDs * (tD-1)
        n_groups = min(B_all.size()[1] - span, B.size()[1])
        if n_groups > 0:
            for t in range(0, n_groups, tD):
                skip = B_all[:, (-span-t-1):-t:tDs].contiguous() \
                    if t != 0 else B_all[:, -span-1::tDs].contiguous()

                B_skipped[s] = torch.cat([B_skipped[s], skip]) \
                    if B_skipped[s] is not None else skip
    max_prev_frames = tD ** (t_scales-1) * (tD-1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped


def get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B, flow_ref,
                      conf_ref, t_scales, tD):
    '''get temporally subsampled frames for flows'''
    flow_ref_skipped, conf_ref_skipped = [None] * t_scales, [None] * t_scales
    flow_ref_all, flow = get_skipped_frames(flow_ref_all, flow_ref, 1, tD)
    conf_ref_all, conf = get_skipped_frames(conf_ref_all, conf_ref, 1, tD)
    if flow[0] is not None:
        flow_ref_skipped[0], conf_ref_skipped[0] \
            = flow[0][:, 1:], conf[0][:, 1:]

    for s in range(1, t_scales):
        if real_B[s] is not None and real_B[s].size()[1] == tD:
            flow_ref_skipped[s], conf_ref_skipped[s] \
                = flowNet(real_B[s][:, 1:], real_B[s][:, :-1])

    return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped


def get_model(is_train, other_opts=None):
    def Model(config):
        if other_opts is None:
            opts = TrainOptions() if is_train else TestOptions()
        else:
            opts = other_opts
        opts.label_nc = 25
        opts.input_nc = 4
        opts.output_nc = 4
        # opts.no_vgg = True
        # opts.no_ganFeat = True
        # opts.num_D = 1
        # opts.n_layers_D = 1
        # opts.n_gpus_gen = 2
        opts.batchSize = config['batch_size']
        opts.n_frames_G = config['n_ts']
        cvd = os.environ.get('V2V_GPUID', '0')
        opts.gpu_ids = [int(d) for d in cvd.split(',')]
        opts.name = 'vid2vid'
        opts.checkpoints_dir = P.checkpoints
        opts.no_first_img = True
        # opts.use_real_img = False

        opts.n_frames_G = 2

        if is_train:
            G_, D_, F_ = create_model(opts)

            class ModelCls(object):
                G = G_
                D = D_
                F = F_
                opt = opts

            return ModelCls
        else:
            return create_model(opts)
    return Model


TrainModel = get_model(True)
TestModel = get_model(False)


def train_op(model, label, image, inst, save_fake, **kwargs):
    '''Op that is repeated at every iteration.

    Args:
        model (ModelCls): Has three models: Generator Discriminator and
            FlowNet.
        data (dict): Must have the entries `B`, `A`, `inst`.
        save_fake (bool): Makes the model return the generated image.
    '''

    opt = model.opt

    modelG, modelD, flowNet = model.G, model.D, model.F

    # number of gpus used for generator for each batch
    n_gpus = max(1, opt.n_gpus_gen // opt.batchSize)
    tG, tD = opt.n_frames_G, opt.n_frames_D
    tDB = tD * opt.output_nc
    s_scales = opt.n_scales_spatial
    t_scales = opt.n_scales_temporal
    input_nc = 1 if opt.label_nc != 0 else opt.input_nc
    output_nc = opt.output_nc
    label_nc = opt.label_nc

    # n_frames_total = n_frames_load * n_loadings + tG - 1
    _, n_frames_total, _, height, width = image.size()
    # n_frames_total = n_frames_total // opt.output_nc
    # number of total frames loaded into GPU at a time for each batch
    n_frames_load = opt.max_frames_per_gpu * n_gpus
    n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
    # number of loaded frames plus previous frames
    t_len = n_frames_load + tG - 1

    # the last generated frame from previous training batch (which
    # becomes input to the next batch)
    fake_B_last = None
    real_B_all, fake_B_all = None, None
    # all real/generated frames so far
    flow_ref_all, conf_ref_all = None, None,
    # temporally subsampled frames
    real_B_skipped, fake_B_skipped = [None]*t_scales, [None]*t_scales
    # temporally subsampled flows
    flow_ref_skipped, conf_ref_skipped = [None]*t_scales, [None]*t_scales

    logger.debug('n_frames_total: {}, t_len: {}, n_frames_load {}'
                 .format(n_frames_total, t_len, n_frames_load))

    N = 0
    ret_dict = {}
    for i in range(0, n_frames_total-t_len+1, n_frames_load):

        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_A = Variable(label[:, i:(i+t_len), ...])
        input_A = input_A.view(-1, t_len, label_nc, height, width)
        input_B = Variable(image[:, i:(i+t_len), ...])
        input_B = input_B.view(-1, t_len, output_nc, height, width)
        if inst is not None and len(inst.size()) > 2:
            inst_A = Variable(inst[:, i:i+t_len, ...])
            inst_A = inst_A.view(-1, t_len, 1, height, width)
        else:
            inst_A = None

        logger.debug('inA: {}'.format(input_A.size()))
        logger.debug('inB: {}'.format(input_B.size()))
        logger.debug('instA: {}'.format(inst_A if inst_A is None
                                        else inst_A.size()))

        # generator
        rets = modelG(input_A, input_B, inst_A, fake_B_last)
        (fake_B,
         fake_B_raw,
         flow,
         weight,
         real_A,
         real_Bp,
         fake_B_last) = rets

        if i == 0:
            # the first generated image in this sequence
            fake_B_first = fake_B[0, 0]

        # sizes = []
        # for ar in list(rets):
        #     if isinstance(ar, list):
        #         s = '{}x{}'.format(len(ar), ar[0].size())
        #     else:
        #         s = '{}'.format(ar.size())
        #     sizes += [s]

        # msg = 'Generated: \nfake_B: {}\nfake_B_raw: {}\nflow: {}\nweight: {}' \
        #       '\nreal_A: {}\nrealBp: {}\nfake_B_last: {}\n\n\n'
        # msg = msg.format(*sizes)

        # logger.debug(msg)

        # the collection of previous and current real frames
        real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]

        # discriminator
        # individual frame discriminator
        # reference flows and confidences
        flow_ref, conf_ref = flowNet(real_B[:, :, :3, :, :],
                                     real_B_prev[:, :, :3, :, :])
        fake_B_prev = real_B_prev[:, 0:1] \
            if fake_B_last is None else fake_B_last[0][:, -1:]

        if fake_B.size()[1] > 1:
            fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()],
                                    dim=1)

        losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A,
                                    real_B_prev, fake_B_prev, flow, weight,
                                    flow_ref, conf_ref]))
        losses = [torch.mean(x) if x is not None else 0 for x in losses]
        loss_dict = dict(zip(modelD.loss_names, losses))

        # temporal discriminator
        loss_dict_T = []
        # get skipped frames for each temporal scale
        if t_scales > 0:
            real_B_all, real_B_skipped \
                = get_skipped_frames(real_B_all, real_B, t_scales, tD)
            fake_B_all, fake_B_skipped \
                = get_skipped_frames(fake_B_all, fake_B, t_scales, tD)

            flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped \
                = get_skipped_flows(flowNet,
                                    flow_ref_all,
                                    conf_ref_all,
                                    real_B_skipped,
                                    flow_ref,
                                    conf_ref,
                                    t_scales, tD)

        # run discriminator for each temporal scale
        for s in range(t_scales):
            if real_B_skipped[s] is not None \
                    and real_B_skipped[s].size()[1] == tD:
                losses = modelD(s+1,
                                [real_B_skipped[s],
                                 fake_B_skipped[s],
                                 flow_ref_skipped[s],
                                 conf_ref_skipped[s]])
                losses = [torch.mean(x) if not isinstance(x, int) else x
                          for x in losses]
                loss_dict_T.append(dict(zip(modelD.loss_names_T,
                                            losses)))

        # collect losses
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] \
            + loss_dict['G_VGG']
        loss_G += loss_dict['G_Warp'] + loss_dict['F_Flow'] \
            + loss_dict['F_Warp'] + loss_dict['W']
        if opt.add_face_disc:
            loss_G += loss_dict['G_f_GAN'] + loss_dict['G_f_GAN_Feat']
            loss_D += (loss_dict['D_f_fake'] + loss_dict['D_f_real']) * 0.5

        # collect temporal losses
        loss_D_T = []
        t_scales_act = min(t_scales, len(loss_dict_T))
        for s in range(t_scales_act):
            loss_G += loss_dict_T[s]['G_T_GAN'] \
                + loss_dict_T[s]['G_T_GAN_Feat'] \
                + loss_dict_T[s]['G_T_Warp']
            loss_D_T.append((loss_dict_T[s]['D_T_fake']
                             + loss_dict_T[s]['D_T_real']) * 0.5)

        # Backward Pass
        optimizer_G = modelG.optimizer_G
        optimizer_D = modelD.optimizer_D
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        # individual frame discriminator
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        # temporal discriminator
        for s in range(t_scales_act):
            optimizer_D_T = getattr(modelD, 'optimizer_D_T'+str(s))
            optimizer_D_T.zero_grad()
            loss_D_T[s].backward()
            optimizer_D_T.step()

        # logger.debug('loss_dict_T: {}\n\n\n\n\n'.format(loss_D_T))

        def det(t):
            return t.detach()

        walk(loss_dict, det, inplace=True)
        walk(loss_dict_T, det, inplace=True)

        gen = [fake_B.detach()] if i == 0 else [fake_B.detach()]
        ret_dict_t = {'losses': {'per_frame': loss_dict,
                                 'temporal': loss_dict_T},
                      'generated': gen,
                      'images': image,
                      'label': label,
                      'real_A': [real_A.detach()],
                      'real_B': [real_B.detach()]}

        if i == 0:
            ret_dict = ret_dict_t
        else:
            def sums(key, t):
                other = retrieve(key, ret_dict_t['losses'])
                t += other
                return t

            ret_dict['losses'] = walk(ret_dict_t['losses'],
                                      sums,
                                      pass_key=True)

            ret_dict['generated'] += ret_dict_t['generated']
            ret_dict['real_A'] += ret_dict_t['real_A']
            ret_dict['real_B'] += ret_dict_t['real_B']

        N += 1

    def div(t):
        t /= N
        return t

    walk(ret_dict['losses'], div, inplace=True)

    logger.debug('generated: {}'.format(len(ret_dict['generated'])))
    logger.debug('real_A:    {}'.format(len(ret_dict['real_A'])))
    logger.debug('real_B:    {}'.format(len(ret_dict['real_B'])))

    ret_dict['generated'] = torch.cat(ret_dict['generated'], 1)
    ret_dict['real_A'] = torch.cat(ret_dict['real_A'], 1)
    ret_dict['real_B'] = torch.cat(ret_dict['real_B'], 1)

    G = ret_dict['generated']
    A = ret_dict['real_A']
    B = ret_dict['real_B']

    logger.debug('generated: {}, {}, {}'.format(G.size(), G.type(), [G.min(), G.max()]))
    logger.debug('real_A:    {}, {}, {}'.format(A.size(), A.type(), [A.min(), A.max()]))
    logger.debug('real_B:    {}, {}, {}'.format(B.size(), B.type(), [B.min(), B.max()]))

    return ret_dict


def test_op(model, label, image, inst, change_seq, save_fake, **kwargs):

    if change_seq:
        model.fake_B_prev = None

    logger.info('im {}'.format(image.shape))

    generated = model.inference(label, image, inst)

    real_A = generated[1]
    real_B = generated[2]
    fake_B = generated[0]

    return {'generated': fake_B,
            'images': image,
            'label': label,
            'real_A': real_A,
            'real_B': real_B}


def flow_op(model, label, image, inst, change_seq, save_fake, **kwargs):
    N_ts = image.size()[1]
    logger.info('N_ts {}'.format(N_ts))
    labels = [label[:, i:i+2] for i in range(N_ts-2)]
    images = [image[:, i:i+2] for i in range(N_ts-2)]
    insts = inst if inst is not None else [inst] * len(labels)
    change_seqs = (len(labels) - 1) * [False] + [True]
    save_fakes = [False] * len(labels)

    logger.info('ims {}'.format(image.size()))
    logger.info('ims {}'.format(len(images)))

    gathered_dict = {}
    iterator = zip(labels, images, insts, change_seqs, save_fakes)
    for frame, [l, i, inst, c, s] in enumerate(iterator):

        ret_dict = test_op(model, l, i, inst, c, s)

        for k, v in ret_dict.items():
            if k not in gathered_dict:
                gathered_dict[k] = []
            gathered_dict[k] += [v]
            logger.info('{} - {}: {}'.format(frame, k, v.size() if hasattr(v, 'size') else v))

    for k, v in gathered_dict.items():
        gathered_dict[k] = torch.cat(v)

    return gathered_dict


class ToNumpyHook(Hook):
    def after_step(self, step, results):
        def convert(var_or_tens):
            if hasattr(var_or_tens, 'cpu'):
                var_or_tens = var_or_tens.cpu()

            if isinstance(var_or_tens, torch.autograd.Variable):
                return var_or_tens.data.numpy()
            elif isinstance(var_or_tens, torch.Tensor):
                return var_or_tens.numpy()
            else:
                return var_or_tens

        walk(results, convert, inplace=True)


class PlotImageBatch(Hook):
    def __init__(self, root, keys, names=None, time_axis=None):
        '''Extracts the keys from the results, and plots the resulting tensor.

        Args:
            root (str): path/to/where the images are saved.
            keys (list of str): key/to/image_tensor.
            names (list of str): names for the image batches for saveing. If
                None, the laste element of keys.split('/') is used.
            time_axis (int): If given, this axis is used to split the image
                batches into single frame batches. These must then have a
                rank of 4.
        '''

        self.root = root
        self.keys = keys
        self.names = names
        if self.names is not None:
            assert len(self.names) == len(self.keys)

        self.time_axis = time_axis

        self.logger = get_logger(self)

    def after_step(self, batch_index, results):
        step = retrieve('global_step', results)

        for i, key in enumerate(self.keys):
            image_batch = retrieve(key, results)
            if self.names is not None:
                name = self.names[i]
            else:
                name = key.split('/')[-1]

            if self.time_axis is not None:
                n = image_batch.shape[self.time_axis]
                t_batches = np.split(image_batch, n, axis=self.time_axis)
                for ts, sub_batch in enumerate(t_batches):
                    sub_batch = sub_batch.squeeze(self.time_axis)
                    savename = name + '_{:0>7d}-{:0>4}-ts{:0>2}.png' \
                        .format(step, batch_index, ts)
                    savename = os.path.join(self.root, savename)
                    plot_batch(sub_batch, savename)
            else:
                savename = name + '_{:0>7d}-{:0>4}.png'.format(step,
                                                               batch_index)
                savename = os.path.join(self.root, savename)
                plot_batch(image_batch, savename)


class PrepareV2VDataHook(Hook):
    def __init__(self, isTrain=True):
        self.logger = get_logger(self)
        self.last_fid = -1
        self.isTrain = isTrain

    def before_step(self, step, fetches, feeds, batch):
        # Data comes from a Sequence Dataset
        im_heat = {}
        for key in ['target', 'heatmaps']:
            val = feeds[key]
            self.logger.info('{}: {}'.format(key, np.shape(val)))
            if len(val.shape) == 4:
                val = np.expand_dims(val, 0)
            val = np.transpose(val, [0, 1, 4, 2, 3])
            im_heat[key] = val
        images = im_heat['target']
        heatmaps = im_heat['heatmaps']

        current_fid = feeds['fid'][0]
        if len(feeds['fid']) == 2:
            current_fid = current_fid[0]
        self.logger.info('cfid {}'.format(current_fid))

        feeds['label'] = torch.from_numpy(heatmaps).float()
        feeds['image'] = torch.from_numpy(images).float()
        feeds['inst'] = None
        feeds['feat'] = None
        if len(current_fid) > 1:
            feeds['change_seq'] = False
        else:
            feeds['change_seq'] = not ((self.last_fid + 1) == current_fid)
        feeds['save_fake'] = False

        self.last_fid = current_fid

    def after_step(self, step, results):
        for result in ['generated', 'label', 'images', 'real_A', 'real_B']:
            if result in results['step_ops'][0]:
                images = results['step_ops'][0][result].float()
                if len(images.size()) == 5:
                    # Transpose [0, 1, 2, 3, 4] to [0, 1, 3, 4, 2]
                    images = images.transpose(2, 4)  # [0, 1, 4, 3, 2]
                    images = images.transpose(2, 3)  # [0, 1, 3, 4, 2]

                    if images.size()[4] == 25:
                        images = images.mean(4, keepdim=True)

                    if not self.isTrain:
                        images = images[:, -1, ...]
                elif len(images.size()) == 4:
                    # Transpose [0, 1, 2, 3] to [0, 2, 3, 1]
                    images = images.transpose(1, 3)  # [0, 3, 2, 1]
                    images = images.transpose(1, 2)  # [0, 2, 3, 1]

                    if images.size()[3] == 25:
                        images = images.mean(3, keepdim=True)

                elif len(images.size()) == 3:
                    # Transpose [0, 1, 2] to [1, 2, 0]
                    images = images.transpose(0, 2)  # [2, 1, 0]
                    images = images.transpose(0, 1)  # [1, 2, 0]

                    if images.size()[2] == 25:
                        images = images.mean(2, keepdim=True)

                    if not self.isTrain:
                        images = images.unsqueeze(0)

                results['step_ops'][0][result] = images


class V2VTrainer(PyHookedModelIterator):

    def __init__(self,
                 config,
                 root,
                 model,
                 hook_freq=1,
                 num_epochs=100,
                 hooks=[],
                 bar_position=0):

        super().__init__(config,
                         root,
                         model,
                         hook_freq,
                         num_epochs,
                         hooks,
                         bar_position,
                         desc='Train')

        image_names = [
                'step_ops/0/generated',
                'step_ops/0/images',
                'step_ops/0/label',
                'step_ops/0/real_A',
                'step_ops/0/real_B'
                ]

        loss_names = ['G_VGG',
                      'G_GAN',
                      'G_GAN_Feat',
                      'D_real',
                      'D_fake',
                      'G_Warp',
                      'F_Flow',
                      'F_Warp',
                      'W']

        loss_names_T = ['G_T_GAN',
                        'G_T_GAN_Feat',
                        'D_T_real',
                        'D_T_fake',
                        'G_T_Warp']

        opt = model.opt

        prefix = 'step_ops/0/losses/per_frame/'
        scalar_names = [prefix + n for n in loss_names]

        prefix_t = 'step_ops/0/losses/temporal/'
        for s in range(opt.n_frames_G - 1):
            s = '{}/'.format(s)
            scalar_names += [prefix_t + s + n for n in loss_names_T]

        ImPlotHook = IntervalHook([PlotImageBatch(P.latest_eval,
                                                  image_names,
                                                  time_axis=1)],
                                  interval=10,
                                  max_interval=500,
                                  modify_each=10)

        checks = []
        models = [model.G, model.D, model.F]
        names = ['gen', 'discr', 'flow']
        for n, m in zip(names, models):
            checks += [PyCheckpointHook(P.checkpoints,
                                        m,
                                        'v2v_{}'.format(n),
                                        interval=config['ckpt_freq'])]

        self.hook_freq = 1
        self.hooks += [PrepareV2VDataHook()]
        self.hooks += checks
        self.hooks += [ToNumpyHook(),
                       ImPlotHook,
                       PyLoggingHook(scalar_keys=scalar_names,
                                     log_keys=scalar_names,
                                     root_path=P.latest_eval,
                                     interval=config['log_freq']),
                       IncreaseLearningRate(model, opt.niter)]

    def step_ops(self):
        return [train_op]

    def initialize(self, checkpoint_path=None):
        if checkpoint_path is not None:
            self.logger.info(checkpoint_path)

            if 'gen' in checkpoint_path or 'discr' in checkpoint_path or 'flow' in checkpoint_path:
                base_path = '_'.join(checkpoint_path.split('_')[:-1])
            else:
                base_path = checkpoint_path

            self.model.G.load_state_dict(torch.load(base_path + '_gen.ckpt'))
            self.model.F.load_state_dict(torch.load(base_path + '_flow.ckpt'))
            self.model.D.load_state_dict(torch.load(base_path + '_discr.ckpt'))

    def fit(self, *args, **kwargs):
        return self.iterate(*args, **kwargs)


class Vid2VidEvaluator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        kwargs['desc'] = 'Eval'
        kwargs['hook_freq'] = 1
        super().__init__(*args, **kwargs)

        image_names = [
                'step_ops/0/generated',
                'step_ops/0/images',
                'step_ops/0/label',
                'step_ops/0/real_A',
                'step_ops/0/real_B'
                ]

        M_reid = MetricTuple({},
                             {'image': image_names[-1],
                              'generated': image_names[0]},
                             reIdMetricFn(nogpu=True),
                             'reId-distance')

        M_ssim = MetricTuple({},
                             {'batch1': image_names[-1],
                              'batch2': image_names[0]},
                             ssim_metric,
                             'ssim')

        M_l2 = MetricTuple({},
                           {'batch1': image_names[-1],
                            'batch2': image_names[0]},
                           l2_metric,
                           'l2')

        # M_kp = MetricTuple({},
        #                    {'batch1': image_names[1],
        #                     'batch2': image_names[0]},
        #                    kp_metric(),
        #                    'kp_distance')

        Ms = [M_reid, M_ssim, M_l2]  # , M_kp]

        MHook = MetricHook(Ms, P.latest_eval)

        def filter_fn(name):
            return '_gen' in name

        restore = RestorePytorchModelHook(self.model,
                                          P.checkpoints,
                                          filter_fn,
                                          self.set_global_step)

        self.hooks = [WaitForCheckpointHook(P.checkpoints,
                                            filter_fn,
                                            callback=restore),
                      PrepareV2VDataHook(isTrain=False),
                      ToNumpyHook(),
                      MHook,
                      PlotImageBatch(P.latest_eval,
                                     image_names),
                      KeepBestCheckpoints(P.checkpoints,
                                          '{:0>6d}_metrics.npz',
                                          'ssim',
                                          n_keep=1)]

    def step_ops(self):
        return [test_op]


class IncreaseLearningRate(Hook):
    '''Invokes the increment of the learning rate.'''
    def __init__(self, model, niter):
        self.model = model
        self.niter = niter
        self.logger = get_logger(self)

    def after_epoch(self, epoch, *args, **kwargs):
        if epoch > self.niter:
            self.logger.info('updating learning rate')
            model.G.update_learning_rate(epoch)
            model.D.update_learning_rate(epoch)


class DummyHook(Hook):
    def after_step(self, step, results):
        results["step_ops"] = results["step_ops"][0]
        results["step_ops"]["params"] = results["step_ops"]["generated"]


class StoreImageSequence(Hook):
    def __init__(self, root, keys, names=None):
        '''Extracts all images in a sequence and stores them at root.'''

        self.root = root
        self.keys = keys
        self.names = names

        if self.names is not None:
            assert len(self.names) == len(self.keys)

    def before_step(self, step, fetches, feeds, batch):
        self.indices = batch['index_']
        self.boxes = batch['box']
        self.paths = batch['file_id']

    def after_step(self, batch_index, results):
        step = retrieve('global_step', results)

        eval_dir = os.path.join(self.root, '{:6>0}'.format(step))

        os.makedirs(eval_dir, exist_ok=True)

        for key in self.keys:
            if self.names is not None:
                name = self.names[i]
            else:
                name = key.split('/')[-1]

            batched_sequences = retrieve(key, results)
            if key.split('/')[-1] == 'images':
                batched_sequences = batched_sequences[1:]

            iterator = zip(self.indices,
                           self.boxes,
                           self.paths,
                           batched_sequences)

            logger.info('seq {}'.format(np.shape(batched_sequences)))
            for frame, [idx, b, fid, image] in enumerate(iterator):
                logger.info('fid {}'.format(fid))
                logger.info('b {}'.format(b))
                logger.info('seq {}'.format(image.shape))
                savename = os.path.join(eval_dir,
                                        '{:0>7}_{}-{:0>3}.png'.format(idx, name, frame))
                save_image(image, savename)
                np.save(savename.replace('.png', '-box.npy'), b[frame])
                with open(savename.replace('.png', '-org.txt'), 'w+') as f:
                    f.write(fid[0])


class ImageEvaluator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        kwargs['desc'] = 'Eval'
        kwargs['hook_freq'] = 1
        super().__init__(*args, **kwargs)

        restore_callback = RestorePytorchModelHook(self.model,
                                                   P.checkpoints,
                                                   self._global_step)

        self.hooks += [WaitForCheckpointHook(P.checkpoints,
                                             callback=restore_callback),
                       PrepareV2VDataHook(),
                       ToNumpyHook()]
        self.hooks += [DummyHook(), TransferHook(self.root, self.model, self.config.get("eval_video", True))]

    def step_ops(self):
        return [test_op]


class SequenceEvaluator(PyHookedModelIterator):
    def __init__(self, *args, **kwargs):
        kwargs['desc'] = 'Eval'
        kwargs['hook_freq'] = 1
        super().__init__(*args, **kwargs)

        restore_callback = RestorePytorchModelHook(self.model,
                                                   P.checkpoints,
                                                   self._global_step)

        self.hooks += [WaitForCheckpointHook(P.checkpoints,
                                             lambda c: '_gen' in c,
                                             callback=restore_callback),
                       PrepareV2VDataHook(),
                       ToNumpyHook()]
        self.hooks += [StoreImageSequence(P.latest_eval,
                                         ['step_ops/0/generated',
                                          'step_ops/0/images'])]

    def step_ops(self):
        return [flow_op]


if __name__ == '__main__':
    init_project('logs', 'vid2vid', postfix='testing')
    logger = get_logger('vid2vid_impl')
    P = ProjectManager()

    opt = TrainOptions()
    opt.label_nc = nl = 25
    opt.input_nc = 4
    opt.output_nc = nc = 4
    opt.n_frames_G = tG = 3
    opt.no_vgg = True
    cvd = os.environ.get('V2V_GPUID', '0')
    opt.gpu_ids = [int(d) for d in cvd.split(',')]
    opt.name = 'pix2pixHD'
    opt.checkpoints_dir = P.checkpoints

    model = get_model(True, opt)
    model = model('asd')

    # data has shape [bs, n_frames, n_ch, x, y]
    im = np.ones([16, tG, nc, 128, 128])
    heat = np.ones([16, tG, nl, 128, 128])
    data = {
            'input_B': Variable(torch.from_numpy(im).float()),
            'input_A': Variable(torch.from_numpy(heat).float()),
            'inst_A': None,
            'fake_B_prev': None
            }

    print(model.G)
    print(model.D)
    print(model.F)
    ret = model.G(**data)

    def fn(val):
        print(val.size())

    walk(list(ret), fn)
    print('This works!')
