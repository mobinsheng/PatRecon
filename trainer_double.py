from torch.autograd import Variable
from utils import AverageMeter
from loss import *
from v2_double_net import X2CTGenerator, CT2XGenerator
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer_ReconNet_Double(nn.Module):
    def __init__(self, args):
        super(Trainer_ReconNet_Double, self).__init__()

        self.exp_name = args.exp
        self.arch = args.arch
        self.print_freq = args.print_freq
        self.output_path = args.output_path
        self.resume = args.resume
        self.best_loss = 1e5
        self.device = torch.device("cpu")
        self.print_model = False
        self.use_compose_loss = False
        self.enable_adaptive_lr = args.enable_adaptive_lr
        self.enable_norm = args.enable_norm

        # create model
        print("=> Creating model...")
        if self.arch == 'ReconNet':
            """self.model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel, gain=args.init_gain,
                                  init_type=args.init_type)"""
            self.x2ct_model = X2CTGenerator(input_channels=args.num_views, output_channels=args.output_channel)
            self.x2ct_model = nn.DataParallel(self.x2ct_model).to(self.device)

            self.ct2x_model = CT2XGenerator(input_channels=args.num_views, output_channels=args.output_channel)
            self.ct2x_model = nn.DataParallel(self.ct2x_model).to(self.device)
        else:
            assert False, print('Not implemented model: {}'.format(self.arch))

        # define loss function
        if args.loss == 'l1':
            # L1 loss
            self.criterion = nn.L1Loss(size_average=True, reduce=True).to(self.device)
        elif args.loss == 'l2':
            # L2 loss (mean-square-error)
            self.criterion = nn.MSELoss(reduction='mean').to(self.device)
        else:
            assert False, print('Not implemented loss: {}'.format(args.loss))

        if self.use_compose_loss:
            self.criterion = CompositeLoss().to(
                self.device)  # Eagle_Loss(patch_size=3).to(self.device)#CompositeLoss().to(self.device)

        # define optimizer
        if args.optim == 'adam':
            self.x2ct_optimizer = torch.optim.Adam(self.x2ct_model.parameters(),
                                                   lr=args.lr,
                                                   betas=(0.5, 0.999),
                                                   weight_decay=args.weight_decay,
                                                   )
            self.ct2x_optimizer = torch.optim.Adam(self.ct2x_model.parameters(),
                                                   lr=args.lr,
                                                   betas=(0.5, 0.999),
                                                   weight_decay=args.weight_decay,
                                                   )
        else:
            assert False, print('Not implemented optimizer: {}'.format(args.optim))

        if self.enable_adaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.x2ct_optimizer, gamma=0.9)

    def train_epoch(self, train_loader, epoch):

        if not self.print_model:
            self.print_model = True
            print(self.x2ct_model)
            print(self.ct2x_model)

        train_loss = AverageMeter()

        # train mode
        self.x2ct_model.train()
        self.ct2x_model.train()

        for i, (input, target) in enumerate(train_loader):

            input_var, target_var = Variable(input), Variable(target)
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)

            # compute output
            x2ct_output = self.x2ct_model(input_var)
            ct_for_input = torch.unsqueeze(x2ct_output, 1)
            ct2x_output = self.ct2x_model(ct_for_input)
            # print("target:", target.shape)
            # print("output:", output.shape)

            # compute loss
            # loss = self.criterion(output, target_var)
            x2ct_loss = self.criterion(x2ct_output, target_var)
            ct2x_loss = self.criterion(ct2x_output, input)

            if self.enable_norm:
                x2ct_loss *= 100.0
                ct2x_loss *= 100.0
                pass

            train_loss.update(x2ct_loss.data.item() + ct2x_loss.data.item(), input.size(0))

            self.ct2x_optimizer.zero_grad()
            ct2x_loss.backward(retain_graph=True)
            self.ct2x_optimizer.step()

            # compute gradient and do SGD step
            self.x2ct_optimizer.zero_grad()
            x2ct_loss.backward()
            self.x2ct_optimizer.step()

            """self.ct2x_optimizer.zero_grad()

            ct2x_loss.backward(retain_graph=True)
            self.ct2x_optimizer.step()

            self.ct2x_optimizer.zero_grad()
            self.x2ct_optimizer.zero_grad()

            x2ct_loss.backward()
            self.x2ct_optimizer.step()"""

            # display info
            if i % self.print_freq == 0:
                print('Epoch: [{0}] \t'
                      'Iter: [{1}/{2}]\t'
                      'x2ct_loss: [{3}]  ct2x_loss: [{4}]\t'
                      'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'
                .format(
                    epoch, i, len(train_loader),
                    x2ct_loss.data.item(), ct2x_loss.data.item(),
                    loss=train_loss))

        # finish current epoch
        print('Finish Epoch: [{0}]\t'
              'Average Train Loss: {loss.avg:.5f}\t'.format(
            epoch, loss=train_loss))

        return train_loss.avg

    def validate(self, val_loader):

        val_loss = AverageMeter()
        batch_time = AverageMeter()

        last_output = None

        # evaluation mode
        self.x2ct_model.eval()
        self.ct2x_model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input), Variable(target)
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)

            # compute output
            x2ct_output = self.x2ct_model(input_var)
            ct_for_input = torch.unsqueeze(x2ct_output, 1)
            ct2x_output = self.ct2x_model(ct_for_input)

            last_output = x2ct_output

            # compute loss
            # loss = self.criterion(output, target_var)
            x2ct_loss = self.criterion(x2ct_output, target_var)
            ct2x_loss = self.criterion(ct2x_output, input)

            if self.enable_norm:
                x2ct_loss *= 100.0
                ct2x_loss *= 100.0
                pass

            val_loss.update(x2ct_loss.data.item() + ct2x_loss.data.item(), input.size(0))

            if self.enable_adaptive_lr:
                self.scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'x2ct_loss: [{2}]  ct2x_loss: [{3}]\t'
                  'Time {batch_time.val: .3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                i, len(val_loader),
                x2ct_loss.data.item(), ct2x_loss.data.item(),
                batch_time=batch_time,
                loss=val_loss))

        return val_loss.avg, last_output

    def save(self, curr_val_loss, epoch):
        # update best loss and save checkpoint
        is_best = curr_val_loss < self.best_loss
        self.best_loss = min(curr_val_loss, self.best_loss)

        state = {'epoch': epoch + 1,
                 'arch': self.arch,
                 'state_dict': self.x2ct_model.state_dict(),
                 'state_dict2': self.ct2x_model.state_dict(),
                 'best_loss': self.best_loss,
                 'optimizer': self.x2ct_optimizer.state_dict(),
                 'optimizer2': self.ct2x_optimizer.state_dict(),
                 }

        filename = osp.join(self.output_path, 'curr_model.pth.tar')
        best_filename = osp.join(self.output_path, 'best_model.pth.tar')

        print('! Saving checkpoint: {}'.format(filename))
        torch.save(state, filename)

        if is_best:
            print('!! Saving best checkpoint: {}'.format(best_filename))
            shutil.copyfile(filename, best_filename)

    def load(self):

        if self.resume == 'best':
            ckpt_file = osp.join(self.output_path, 'best_model.pth.tar')
        elif self.resume == 'final':
            ckpt_file = osp.join(self.output_path, 'curr_model.pth.tar')
        else:
            assert False, print("=> no available checkpoint '{}'".format(ckpt_file))

        if osp.isfile(ckpt_file):
            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file)
            start_epoch = checkpoint['epoch']

            self.best_loss = checkpoint['best_loss']
            self.x2ct_model.load_state_dict(checkpoint['state_dict'])
            self.ct2x_model.load_state_dict(checkpoint['state_dict2'])
            self.x2ct_optimizer.load_state_dict(checkpoint['optimizer'])
            self.ct2x_optimizer.load_state_dict(checkpoint['optimizer2'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))

        return start_epoch
