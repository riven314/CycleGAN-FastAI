from pdb import set_trace
from fastai.vision import *
from fastai.callbacks.tensorboard import LearnerTensorboardWriter, ImageTBWriter

class CycleGANTrainer(LearnerCallback):
    _order = -20 #Need to run before the Recorder
    
    def _set_trainable(self, D_A=False, D_B=False):
        gen = (not D_A) and (not D_B)
        requires_grad(self.learn.model.G_A, gen)
        requires_grad(self.learn.model.G_B, gen)
        requires_grad(self.learn.model.D_A, D_A)
        requires_grad(self.learn.model.D_B, D_B)
        if not gen:
            self.opt_D_A.lr, self.opt_D_A.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_A.wd, self.opt_D_A.beta = self.learn.opt.wd, self.learn.opt.beta
            self.opt_D_B.lr, self.opt_D_B.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D_B.wd, self.opt_D_B.beta = self.learn.opt.wd, self.learn.opt.beta
    
    def on_train_begin(self, **kwargs):
        self.G_A,self.G_B = self.learn.model.G_A,self.learn.model.G_B
        self.D_A,self.D_B = self.learn.model.D_A,self.learn.model.D_B
        self.crit = self.learn.loss_func.crit
        if not getattr(self,'opt_G',None):
            self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G_A), *flatten_model(self.G_B))])
        else: 
            self.opt_G.lr,self.opt_G.wd = self.opt.lr,self.opt.wd
            self.opt_G.mom,self.opt_G.beta = self.opt.mom,self.opt.beta
        if not getattr(self,'opt_D_A',None):
            self.opt_D_A = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_A))])
        if not getattr(self,'opt_D_B',None):
            self.opt_D_B = self.learn.opt.new([nn.Sequential(*flatten_model(self.D_B))])
        self.learn.opt.opt = self.opt_G.opt
        self._set_trainable()
        # identity loss, gen loss, cycle loss (generator)
        self.id_smter,self.gen_smter,self.cyc_smter = SmoothenValue(0.98),SmoothenValue(0.98),SmoothenValue(0.98)
        # discriminator A and B loss (discriminator)
        self.da_smter,self.db_smter = SmoothenValue(0.98),SmoothenValue(0.98)
        self.recorder.add_metric_names(['id_loss', 'gen_loss', 'cyc_loss', 'D_A_loss', 'D_B_loss'])
        
    def on_batch_begin(self, last_input, **kwargs):
        self.learn.loss_func.set_input(last_input)
    
    def on_backward_begin(self, **kwargs):
        # right after generator update
        self.id_smter.add_value(self.loss_func.id_loss.detach().cpu())
        self.gen_smter.add_value(self.loss_func.gen_loss.detach().cpu())
        self.cyc_smter.add_value(self.loss_func.cyc_loss.detach().cpu())
    
    def on_batch_end(self, last_input, last_output, **kwargs):
        # for discriminators update
        #set_trace()
        self.G_A.zero_grad(); self.G_B.zero_grad()
        fake_A, fake_B = last_output[0].detach(), last_output[1].detach()
        real_A, real_B = last_input
        # forward pass and backpropagate on discriminator A
        self._set_trainable(D_A=True)
        self.D_A.zero_grad()
        loss_D_A = 0.5 * (self.crit(self.D_A(real_A), True) + self.crit(self.D_A(fake_A), False))
        self.da_smter.add_value(loss_D_A.detach().cpu())
        loss_D_A.backward()
        self.opt_D_A.step()
        # forward pass and backpropagate on discriminator B
        self._set_trainable(D_B=True)
        self.D_B.zero_grad()
        loss_D_B = 0.5 * (self.crit(self.D_B(real_B), True) + self.crit(self.D_B(fake_B), False))
        self.db_smter.add_value(loss_D_B.detach().cpu())
        loss_D_B.backward()
        self.opt_D_B.step()
        # freeze discrimintors and unfreeze generators for generators update
        self._set_trainable()
        
    def on_epoch_end(self, last_metrics, **kwargs):
        # last_metrics is None
        # add_metrics is for updating last_metrics
        set_trace()
        return add_metrics(last_metrics, 
                           [s.smooth for s in [
                               self.id_smter, self.gen_smter, self.cyc_smter, self.da_smter,self.db_smter
                           ]]
                          )
    

class GANTensorboardWriter(LearnerTensorboardWriter):
    """
    Callback for GANLearners that writes to Tensorboard.
    Extends LearnerTensorboardWriter and adds output image writes.
    
    :param:
        hist_iters : iteration to update model weights histogram 
        loss_iters : iteration to update loss value
        stats_iters : iteration to update gradient statistics
        visual_iters : iteration to update generator
    """
    def __init__(self, learn, base_dir, name, loss_iters, hist_iters, stats_iters, visual_iters):
        super().__init__(learn = learn, base_dir = base_dir, name = name, 
                         loss_iters = loss_iters, hist_iters=hist_iters, 
                         stats_iters = stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageTBWriter()
        self.gen_stats_updated = True
        self.crit_stats_updated = True

    def _write_weight_histograms(self, iteration):
        """ 
        Writes model weight histograms to Tensorboard.
        update self.hist_writer (HistogramTBWriter, inherented from LearnerTensorboardWriter)
        """
        G_A, G_B = self.learn.model.G_A, self.learn.model.G_B
        D_A, D_B = self.learn.model.D_A, self.learn.model.D_B
        self.hist_writer.write(model = G_A, iteration = iteration, 
                               tbwriter = self.tbwriter, name = 'G_A_histogram')
        self.hist_writer.write(model = G_B, iteration = iteration,
                               tbwriter = self.tbwriter, name = 'G_B_histogram')
        self.hist_writer.write(model = D_A, iteration = iteration, 
                               tbwriter = self.tbwriter, name = 'D_A_histogram')
        self.hist_writer.write(model = D_B, iteration = iteration,
                               tbwriter = self.tbwriter, name = 'D_B_histogram')

    def _write_gen_model_stats(self, iteration:int):
        """ 
        Writes gradient statistics for generator to Tensorboard.
        update self.stats_writer (ModelStatsTBWriter)
        """
        G_A = self.learn.G_A
        G_B = self.learn.G_B
        self.stats_writer.write(model = G_A, iteration = iteration, 
                                tbwriter = self.tbwriter, name = 'G_A_model_stats')
        self.stats_writer.write(model = G_B, iteration = iteration,
                                tbwriter = self.tbwriter, name = 'G_B_model_stats')
        self.gen_stats_updated = True

    def _write_critic_model_stats(self, iteration):
        """
        Writes gradient statistics for critic to Tensorboard.
        update self.stats_writer (ModelStatsTBWriter)
        """
        D_A = self.learn.D_A
        D_B = self.learn.D_B
        self.stats_writer.write(model = D_A, iteration = iteration, 
                                tbwriter = self.tbwriter, name = 'D_A_model_stats')
        self.stats_writer.write(model = D_B, iteration = iteration,
                                tbwriter = self.tbwriter, name = 'D_B_model_stats')
        self.crit_stats_updated = True

    def _write_model_stats(self, iteration):
        "Writes gradient statistics to Tensorboard."
        # We don't want to write stats when model is not iterated on and hence has zeroed out gradients
        gen_mode = self.learn.gan_trainer.gen_mode
        if gen_mode and not self.gen_stats_updated:
            self._write_gen_model_stats(iteration = iteration)
        if not gen_mode and not self.crit_stats_updated: 
            self._write_critic_model_stats(iteration = iteration)

    def _write_training_loss(self, iteration, last_loss)->None:
        """
        Writes training loss to Tensorboard. 
        note last_loss is a list, NOT scalar
        self.metrics_root = '/metrics/'
        """
        recorder = self.learn.gan_trainer.recorder
        if len(recorder.losses) == 0: return None
        scalar_value = to_np((recorder.losses[-1:])[0])
        
        self.id_smter, self.gen_smter, self.cyc_smter, self.da_smter,self.db_smter
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(
            tag = tag, scalar_value = scalar_value, global_step = iteration
            )

    def _write_images(self, iteration:int)->None:
        "Writes model generated, original and real images to Tensorboard."
        trainer = self.learn.gan_trainer
        #TODO:  Switching gen_mode temporarily seems a bit hacky here.  Certainly not a good side-effect.  Is there a better way?
        gen_mode = trainer.gen_mode
        try:
            trainer.switch(gen_mode=True)
            self.img_gen_vis.write(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
                                    iteration=iteration, tbwriter=self.tbwriter)
        finally: trainer.switch(gen_mode=gen_mode)

    def on_batch_end(self, iteration:int, train:bool, **kwargs)->None:
        "Callback function that writes batch end appropriate data to Tensorboard."
        super().on_batch_end(iteration=iteration, train=train, **kwargs)
        if iteration == 0 and not train: return
        if iteration % self.visual_iters == 0: self._write_images(iteration=iteration)

    def on_backward_end(self, iteration:int, train:bool, **kwargs)->None:
        "Callback function that writes backward end appropriate data to Tensorboard."
        if iteration == 0 and not train: return
        self._update_batches_if_needed()
        #TODO:  This could perhaps be implemented as queues of requests instead but that seemed like overkill. 
        # But I'm not the biggest fan of maintaining these boolean flags either... Review pls.
        if iteration % self.stats_iters == 0: self.gen_stats_updated, self.crit_stats_updated = False, False
        if not (self.gen_stats_updated and self.crit_stats_updated): self._write_model_stats(iteration=iteration)