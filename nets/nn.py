import math

import torch
from torch.nn import functional


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(inplace=True),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, out_ch, s, r, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        if fused:
            features = [Conv(in_ch, r * in_ch, torch.nn.SiLU(True), 3, s),
                        Conv(r * in_ch, out_ch, identity)]
        else:
            features = [Conv(in_ch, r * in_ch, torch.nn.SiLU(True)),
                        Conv(r * in_ch, r * in_ch, torch.nn.SiLU(True), 3, s, r * in_ch),
                        SE(r * in_ch, r),
                        Conv(r * in_ch, out_ch, identity)]
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return self.res(x) + x if self.add else self.res(x)


class EfficientNet(torch.nn.Module):
    def __init__(self, filters) -> None:
        super().__init__()
        feature = [Conv(3, filters[0], torch.nn.SiLU(True), 3, 2),
                   Residual(filters[0], filters[0], 1, 1),
                   Residual(filters[0], filters[0], 1, 1)]
        self.res1 = torch.nn.Sequential(*feature)

        feature = []
        for i in range(4):
            if i == 0:
                feature.append(Residual(filters[0], filters[1], 2, 4))
            else:
                feature.append(Residual(filters[1], filters[1], 1, 4))
        self.res2 = torch.nn.Sequential(*feature)

        feature = []
        for i in range(4):
            if i == 0:
                feature.append(Residual(filters[1], filters[2], 2, 4))
            else:
                feature.append(Residual(filters[2], filters[2], 1, 4))
        self.res3 = torch.nn.Sequential(*feature)

        feature = []
        for i in range(6):
            if i == 0:
                feature.append(Residual(filters[2], filters[3], 2, 4, False))
            else:
                feature.append(Residual(filters[3], filters[3], 1, 4, False))
        for i in range(9):
            if i == 0:
                feature.append(Residual(filters[3], filters[4], 1, 6, False))
            else:
                feature.append(Residual(filters[4], filters[4], 1, 6, False))
        self.res4 = torch.nn.Sequential(*feature)

        initialize_weights(self)

    def forward(self, x):
        feature = []
        x = self.res1(x)
        x = self.res2(x)
        feature.append(x)
        x = self.res3(x)
        x = self.res4(x)
        feature.append(x)
        return feature


class ASPPConv(torch.nn.Sequential):
    def __init__(self, in_ch, out_ch, d):
        modules = [torch.nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                   torch.nn.BatchNorm2d(out_ch),
                   torch.nn.ReLU()]
        super().__init__(*modules)


class ASPPPooling(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = torch.nn.Sequential(torch.nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                       torch.nn.BatchNorm2d(out_ch),
                                       torch.nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = self.res(x.mean((2, 3), keepdim=True))
        return functional.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPModule(torch.nn.Module):
    def __init__(self, in_ch, rates, out_ch):
        super().__init__()
        self.spp1 = torch.nn.Sequential(torch.nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                        torch.nn.BatchNorm2d(out_ch),
                                        torch.nn.ReLU())

        self.spp2 = ASPPConv(in_ch, out_ch, rates[0])
        self.spp3 = ASPPConv(in_ch, out_ch, rates[1])
        self.spp4 = ASPPConv(in_ch, out_ch, rates[2])

        self.spp5 = ASPPPooling(in_ch, out_ch)

        self.project = torch.nn.Sequential(torch.nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
                                           torch.nn.BatchNorm2d(out_ch),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(0.5))

    def forward(self, x):
        res = torch.cat((self.spp1(x),
                         self.spp2(x),
                         self.spp3(x),
                         self.spp4(x),
                         self.spp5(x)), dim=1)
        return self.project(res)


class DeepLabV3(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        filters = [24, 48, 64, 128, 160, 272, 1792]
        num_channels = filters[1] + filters[4]
        self.feature = EfficientNet(filters)
        self.spp_mod = ASPPModule(filters[4], [6, 12, 18], filters[4])
        self.head_fn = torch.nn.Sequential(torch.nn.Conv2d(num_channels, 256, 3, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(256),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Conv2d(256, num_class, 1))
        self.project = torch.nn.Sequential(torch.nn.Conv2d(filters[1], filters[1], 1, bias=False),
                                           torch.nn.BatchNorm2d(filters[1]),
                                           torch.nn.ReLU(inplace=True))

    def forward(self, x):
        low_feature, out_feature = self.feature(x)
        out_feature = self.spp_mod(out_feature)
        out_feature = functional.interpolate(out_feature, low_feature.shape[2:], mode='bilinear', align_corners=False)
        head = torch.cat([low_feature, out_feature], dim=1)
        head = self.head_fn(head)
        return functional.interpolate(head, x.shape[2:], mode='bilinear', align_corners=False)


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def __str__(self) -> str:
        return 'step'

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(torch.optim.Optimizer):
    """
    [https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0.,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss
