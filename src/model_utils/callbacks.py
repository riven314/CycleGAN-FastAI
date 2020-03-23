from pdb import set_trace
from fastai.vision import *
from fastai.callbacks.tensorboard import LearnerTensorboardWriter, ImageTBWriter

class CycleGANTrainer(CycleGANTensorboardWriter):
    _order = -20 #Need to run before the Recorder
    
    def __init__(self, learn, base_dir, name, loss_iters = 25, hist_iters = 500, stats_iters = 100):
        super().__init__(learn, base_dir, name, loss_iters, hist_iters, stats_iters)
        pass
    
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

    def _write_generator_loss(self, iteration):
        for metric_name in ['id_smter', 'gen_smter', 'cyc_smter']:
            val = getattr(self, metric_name)
            tag = self.metrics_root + metric_name
            self.tbwriter.add_scalar(
                tag = tag, scalar_value = scalar_value.detach().numpy(), global_step = iteration
                )

    def _write_critic_loss(self, iteration):
        for metric_name in ['da_smter', 'db_smter']:
            val = getattr(self, metric_name)
            tag = self.metrics_root + metric_name
            self.tbwriter.add_scalar(
                tag = tag, scalar_value = scalar_value.detach().numpy(), global_step = iteration
                )
    
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
    
    def on_backward_begin(self, iteration, **kwargs):
        # right after generator update
        self.id_smter.add_value(self.loss_func.id_loss.detach().cpu())
        self.gen_smter.add_value(self.loss_func.gen_loss.detach().cpu())
        self.cyc_smter.add_value(self.loss_func.cyc_loss.detach().cpu())
        if iteration self._write_generator_loss
    
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


class CycleGANTensorboardWriter(LearnerTensorboardWriter):
    
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