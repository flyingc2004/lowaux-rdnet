from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')        
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training (Note: in train_errnet.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')

        self.parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
        self.parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
        self.parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--synthetic_kernel_sizes', type=str, default='11', help='comma-separated odd blur kernel sizes for synthetic reflections')
        self.parser.add_argument('--low_alpha', type=float, default=1.0, help='minimum synthetic reflection strength')
        self.parser.add_argument('--high_alpha', type=float, default=1.0, help='maximum synthetic reflection strength')
        self.parser.add_argument('--ghost_probability', type=float, default=0.0, help='probability of synthetic double-reflection ghosting')
        self.parser.add_argument('--ghost_max_shift', type=int, default=0, help='maximum synthetic ghosting shift in pixels')
        self.parser.add_argument('--ghost_strength', type=float, default=0.35, help='secondary image weight for synthetic ghosting')
        self.parser.add_argument('--random_reflection_pair', action='store_true', help='randomly pair synthetic reflection sources with clean backgrounds')
        
        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--no_pin_memory', action='store_true', help='disable CUDA pinned-memory DataLoader transfers for stability on constrained or NFS-backed systems')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        # for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan', help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')
        
        # loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg', help='learning rate policy: vgg|mse|ctx|ctx_vgg')
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--pixel_loss', type=str, default='legacy', choices=['legacy', 'charbonnier_gradient'], help='aligned pixel loss configuration')
        self.parser.add_argument('--lambda_gradient', type=float, default=0.2, help='gradient loss weight when using charbonnier_gradient')
        self.parser.add_argument('--lambda_ssim', type=float, default=0.0, help='differentiable SSIM loss weight for aligned samples')

        # architecture experiments. None lets checkpoints restore their saved model configuration.
        self.parser.add_argument('--resblock_dilations', type=str, default=None, help='comma-separated dilation values for the residual blocks')
        self.parser.add_argument('--output_mode', type=str, default=None, choices=['direct', 'reflection_residual'], help='predict transmission directly or subtract a predicted reflection residual')
        self.parser.add_argument('--reset_output_layer', action='store_true', help='zero-initialize the output layer after loading a checkpoint; use only when starting reflection residual training')
        self.parser.add_argument('--refiner_mode', type=str, default=None, choices=['none', 'dilated'], help='optional refinement branch after ERRNet output')
        self.parser.add_argument('--refiner_channels', type=int, default=None, help='hidden channels for the refinement branch')
        self.parser.add_argument('--refiner_dilations', type=str, default=None, help='comma-separated dilations for the refinement branch')
        self.parser.add_argument('--refiner_res_scale', type=float, default=None, help='scale applied to the refinement residual')
        self.parser.add_argument('--freeze_backbone', action='store_true', help='freeze the ERRNet backbone and train only newly added modules')

        # experiment controls. Defaults preserve the original training schedule.
        self.parser.add_argument('--extra_epochs', type=int, default=None, help='train this many epochs after loading the checkpoint')
        self.parser.add_argument('--aligned_fusion_ratios', type=str, default='0.7,0.3', help='synthetic,real paired sampling ratios')
        self.parser.add_argument('--late_aligned_fusion_epoch', type=int, default=45, help='epoch to switch aligned data sampling ratios; negative disables the switch')
        self.parser.add_argument('--late_aligned_fusion_ratios', type=str, default='0.5,0.5', help='late synthetic,real paired sampling ratios')
        self.parser.add_argument('--unaligned_fusion_ratios', type=str, default='0.25,0.5,0.25', help='synthetic,unaligned,real paired sampling ratios')
        self.parser.add_argument('--unaligned_lr_milestones', type=str, default='', help='optional comma-separated absolute epochs for unaligned learning-rate changes')
        self.parser.add_argument('--unaligned_lr_values', type=str, default='', help='comma-separated learning rates matching --unaligned_lr_milestones')
        self.parser.add_argument('--gan_start_epoch', type=int, default=20, help='epoch to enable GAN loss; negative disables GAN loss')
        self.parser.add_argument('--eval_freq', type=int, default=5, help='evaluation frequency in epochs during aligned training')
        
        self.isTrain = True
