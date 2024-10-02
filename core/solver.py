"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
from torchvision.utils import save_image
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics,generate_img
from STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from STEGO.src.crf import dense_crf
from STEGO.src.utils import unnorm, remove_axes, denormalize
from torchvision import transforms

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        #print(self.stego_model)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name, flush=True)
                network.apply(utils.he_init)

        if args.background_separation and args.stego_path != '':
            print("Use stego to perform background separation", flush=True)
            self.stego_model = LitUnsupervisedSegmenter.load_from_checkpoint(args.stego_path).cuda()
        else:
            self.stego_model = None
        

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()


    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(args,loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(args,loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...', flush=True)
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):

            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org, x_mask = inputs.x_src, inputs.y_src, inputs.x_mask
            
            background=None
            # attention
            if args.background_separation:
                background = x_real *(1-x_mask)

                # mask input
                if args.mask_input:
                    x_real= x_real*x_mask

            x_ref, x_ref2, x_ref_mask, x_ref2_mask, y_trg = inputs.x_ref, inputs.x_ref2, inputs.x_ref_mask, inputs.x_ref2_mask, inputs.y_ref
            
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # mask the references
            if args.background_separation and args.mask_reference:
                    x_ref = x_ref * x_ref_mask
                    x_ref2 = x_ref2 * x_ref2_mask
            
            # train the discriminator
            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_mask, background, x_ref=x_ref, x_ref_mask=x_ref_mask, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_mask, background, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            

            # train the generator
            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_mask, background, x_refs=[x_ref, x_ref2], x_ref_masks=[x_ref_mask, x_ref2_mask], masks=masks)
            
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_mask, background, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log, flush=True)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            # if (i+1) % args.eval_every == 0:
            #     calculate_metrics(nets_ema, args, i+1, mode='latent')
            #     calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def debug_image(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
        # fetch random validation images for debugging
        for i in range(10):
            fetcher_val = InputFetcher(args, loaders.val, None, args.latent_dim, 'val')
            inputs_val = next(fetcher_val)
            os.makedirs(args.sample_dir, exist_ok=True)
            filename = args.name + str(i)
            utils.debug_image_for_test(nets_ema, args, inputs=inputs_val, name=filename)

    @torch.no_grad()
    def sample_fid(self):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(self.args.result_dir, exist_ok=True)
        self._load_checkpoint(self.args.resume_iter)
        generate_img(nets_ema, self.args, step=self.args.resume_iter)

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(args,loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(args,loaders.ref, None, args.latent_dim, 'test'))

        filename = args.name + '_reference.jpg'
        fname = ospj(args.result_dir, filename)
        print('Working on {}...'.format(fname), flush=True)
        if args.use_sean_encoder:
            if args.background_separation:
                utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname, x_ref_mask=ref.mask, x_src_mask=src.mask, stego_model=self.stego_model)
            else:
                utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname, x_ref_mask=ref.mask)
        else:
            if args.background_separation:
                utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname, x_ref_mask=ref.mask, x_src_mask=src.mask, stego_model=self.stego_model)
            else:
                utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        # fname = ospj(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')

def compute_d_loss(nets, args, x_real, y_org, y_trg, x_mask, background, z_trg=None, x_ref=None, x_ref_mask=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  
            # sean encoder with mask
            if args.use_sean_encoder:
                s_trg = nets.style_encoder(x_ref, y_trg, x_ref_mask)
            else:
                s_trg = nets.style_encoder(x_ref, y_trg)
        
        x_fake = nets.generator(x_real, s_trg, masks=masks)
        ##########
        ######## REMOVE
        
        #exit()
        # attention
        if args.background_separation:
            x_fake = x_fake*x_mask + background
            
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, x_mask, background, z_trgs=None, x_refs=None, x_ref_masks=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
        x_ref_mask, x_ref2_mask = x_ref_masks

    
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        # sean encoder with mask
        if args.use_sean_encoder:
            s_trg = nets.style_encoder(x_ref, y_trg, x_ref_mask)
        else:
            s_trg = nets.style_encoder(x_ref, y_trg)

    # generate fake image
    x_fake = nets.generator(x_real, s_trg, masks=masks)

    # attention
    if args.background_separation:
        x_fake = x_fake*x_mask  + background
        
    # adversarial loss
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    # sean encoder with mask
    if args.use_sean_encoder:
        s_pred = nets.style_encoder(x_fake, y_trg, x_mask)
    else:
        s_pred = nets.style_encoder(x_fake, y_trg)

    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        # sean encoder with mask
        if args.use_sean_encoder:
            s_trg2 = nets.style_encoder(x_ref2, y_trg, x_ref2_mask)
        else:
            s_trg2 = nets.style_encoder(x_ref2, y_trg)
        
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)

    # attention
    if args.background_separation:
        x_fake2 = x_fake2*x_mask + background

    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None

    # sean encoder with mask
    if args.use_sean_encoder:
        s_org = nets.style_encoder(x_real, y_org, x_mask)
    else:
        s_org = nets.style_encoder(x_real, y_org)

    x_rec = nets.generator(x_fake, s_org, masks=masks)

    # attention
    if args.background_separation:
        x_rec = x_rec*x_mask  + background
    
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    
    if args.visualize_mask and x_refs is not None:
        with torch.no_grad():
            fig, ax = plt.subplots(args.batch_size,7, figsize=(5*5,10))
            for index in range(args.batch_size):
                ax[index,0].imshow(denormalize(x_real[index]).permute(1,2,0).cpu().numpy())
                ax[index,0].set_title("x_real")
                ax[index,1].imshow(x_mask[index].permute(1,2,0).cpu().numpy())
                ax[index,1].set_title("mask")
                ax[index,2].imshow(denormalize(x_ref[index]).permute(1,2,0).cpu())
                ax[index,2].set_title("x_ref")          
                ax[index,3].imshow(denormalize(x_fake[index]).permute(1,2,0).cpu())
                ax[index,3].set_title("x_fake")
                ax[index,4].imshow(denormalize(x_ref2[index]).permute(1,2,0).cpu())
                ax[index,4].set_title("x_ref2")
                ax[index,5].imshow(denormalize(x_fake2[index]).permute(1,2,0).cpu())
                ax[index,5].set_title("x_fake2")
                ax[index,6].imshow(denormalize(x_rec[index]).permute(1,2,0).cpu())
                ax[index,6].set_title("x_rec")
            remove_axes(ax)
            fig.savefig('Mask visualization.png')


    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                      sty=loss_sty.item(),
                      ds=loss_ds.item(),
                      cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


