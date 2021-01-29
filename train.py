"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import numpy as np
from models.networks.sync_batchnorm import DataParallelWithCallback
from util.explanation_utils import explanation_hook, get_explanation, explanation_hook_cifar
# parse options
opt = TrainOptions().parse()


# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        if self.opt.dual:
            from models.pix2pix_dualmodel import Pix2PixModel
        elif self.opt.dual_segspade:
            from models.pix2pix_dual_segspademodel import Pix2PixModel
        elif opt.box_unpair:
            from models.pix2pix_dualunpair import Pix2PixModel
        else:
            from models.pix2pix_model import Pix2PixModel

        self.pix2pix_model = Pix2PixModel(opt)
        self.netG = self.pix2pix_model.netG
        self.discriminator = self.pix2pix_model.netD
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
        # self.d_out = None
        self.explanationType = 'shap'
    def run_generator_one_step(self, data, local_explainable):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        import numpy as np
        d_o1 = self.run_discriminator_one_step(data)
        # print("do1", d_o1)
        d_o1 = np.array(d_o1)
        if local_explainable:
            get_explanation(generated_data=generated, discriminator=self.discriminator, prediction=d_o1,
                            XAItype=self.explanationType, trained_data=data, data_type="abc")
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses, d_out = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses
        # self.d_out = d_out
        return d_out

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr


# create trainer for our model
trainer = Pix2PixTrainer(opt)


# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))
# create tool for visualization
visualizer = Visualizer(opt)

# seq_len_total = 0

# for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
#     print(i)
#     print(data_i['retrival_label_list'].shape)
# #     if i == 1:
# #         break
# exit()
    # label_tensor = data_i['label']
    # label_np = label_tensor.data.cpu().numpy()[0]
    # label_seq = np.unique(label_np)
    # seq_len_total += len(label_seq)
    # break
# print(seq_len_total/float(i))

explanationSwitch = (len(iter_counter.training_epochs()) + 1) / 2 if len(iter_counter.training_epochs()) % 2 == 1 else len(iter_counter.training_epochs()) / 2
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        # local_explainable=True
        iter_counter.record_one_iteration()
        # if (epoch - 1) == explanationSwitch:
        trainer.netG.out.register_backward_hook(explanation_hook)
        local_explainable = True
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, local_explainable)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
