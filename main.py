import os
import random

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

from nets import nn
from utils import config
from utils import util
from utils.dataset import input_fn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device('cuda')
class_names, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))


def train():
    model = nn.DeepLabV3(len(palette)).to(device)

    file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.image_dir))]

    loader, dataset = input_fn(file_names, config.batch_size, palette)

    optimizer = nn.RMSprop(nn.add_weight_decay(model), 0.064 * config.batch_size / 256)
    scheduler = nn.StepLR(optimizer)
    model = torch.nn.DataParallel(model)
    cw = util.get_class_weights(file_names)
    criterion = torch.nn.CrossEntropyLoss(torch.from_numpy(cw)).to(device)
    if not os.path.exists('weights'):
        os.makedirs('weights')
    amp_scale = torch.cuda.amp.GradScaler()
    for epoch in range(config.num_epochs):
        model.train()

        m_loss = torch.zeros(1, device=device)
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(loader), total=len(loader))
        optimizer.zero_grad()
        for i, (images, labels) in progress_bar:
            images = images.to(device, non_blocking=True).float() / 255.0
            labels = labels.to(device, non_blocking=True).long()
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), labels)
            amp_scale.scale(loss).backward()
            amp_scale.step(optimizer)
            amp_scale.update()
            optimizer.zero_grad()

            m_loss = (m_loss * i + loss.detach()) / (i + 1)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, config.num_epochs), *m_loss, mem)
            progress_bar.set_description(s)
        scheduler.step(epoch + 1)
        torch.save({'model': model}, os.path.join('weights', 'model.pt'))


def evaluate():
    model = torch.load(os.path.join('weights', 'model.pt'), device)['model'].float().eval()

    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    img = torch.zeros((1, 3, config.image_size, config.image_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.image_dir))]

    for file_name in file_names:
        image = util.load_image(file_name)
        label = util.load_label(file_name)

        image, label = util.resize(image, label)

        image = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), 0))
        image = image.to(device, non_blocking=True)
        image = image.half() if half else image.float()

        pred = model(image / 255.0)
        pred = np.array(pred.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0))
        pred = util.reverse_one_hot(pred)
        pred = util.colour_code_segmentation(pred, palette)
        pred = np.uint8(pred)
        util.save_images([label, pred], ['TRUE', 'PRED'], os.path.join('results', f'{file_name}.jpg'))


def print_parameters():
    model = nn.DeepLabV3(len(palette)).eval()
    _ = model(torch.zeros(1, 3, config.image_size, config.image_size))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')


if __name__ == '__main__':
    if not os.path.exists('weights'):
        os.makedirs('weights')

    train()
