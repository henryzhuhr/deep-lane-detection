from typing import Tuple
import cv2
import numpy as np
import scipy
import torch, os
import torchvision.transforms as transforms
from data.dataloader import SeqDistributedSampler
from data.dataset import LaneTestDataset
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print, dist_tqdm, is_main_process, synchronize
from data.constant import culane_row_anchor

from utils.img_transform import tensor2cvmat


def main():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    cfg.batch_size = 1

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
        use_aux=False
    ).cuda()                                                            # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7 :]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    output_path = os.path.join(cfg.test_work_dir, cfg.dataset)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()

    loader = get_test_loader(cfg.batch_size, cfg.data_root, cfg.dataset, distributed)
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)          # [B, 201, 18, 4]
        generate_lines(
            out,
            imgs,
            names,
            output_path,
            cfg.griding_num,
            localization_type='rel',
            flip_updown=True,
        )


    # eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed)
def get_test_loader(batch_size, data_root, dataset, distributed):
    img_transforms = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(
            data_root,
            os.path.join(
                data_root,
                'list/train_gt.txt',     # 'list/test.txt',
            ),
            img_transform=img_transforms
        )
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root, os.path.join(data_root, 'test.txt'), img_transform=img_transforms)
        cls_num_per_lane = 56

    if distributed:
        sampler = SeqDistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return loader


COLOR_LIST = [
    (b, g, r) for (color_name, (r, g, b)) in {
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
    }.items()
]


def generate_lines(
    out: torch.Tensor,
    imgs: torch.Tensor,
    names: Tuple[str],       # 文件名，相对于 CULaneROOT 的路径
    output_path: str,
    griding_num,
    localization_type='abs',
    flip_updown=False,
):

    shape = imgs[0, 0].shape
    img_h, img_w = imgs.shape[2 :]
    col_sample = np.linspace(0, shape[1] - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # j batch
    for j in range(out.shape[0]):
        out_j = out[j].data.cpu().numpy()

        if flip_updown:
            out_j = out_j[:, ::-1, :]

        if localization_type == 'abs':
            out_j = np.argmax(out_j, axis=0)
            out_j[out_j == griding_num] = -1
            out_j = out_j + 1
        elif localization_type == 'rel':
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == griding_num] = 0
            out_j = loc
        else:
            raise NotImplementedError

        frame = tensor2cvmat(imgs[j])

        four_lanes = []
        cls_num_per_lane = 18

        row_anchor = culane_row_anchor
        for i in range(out_j.shape[1]):
            color = COLOR_LIST[i]
            lane_i = []
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (
                            int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                            int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                        )
                        cv2.circle(frame, ppp, 5, color, -1)
                        lane_i.append(ppp)
            four_lanes.append(lane_i)

        # for i, lane_i in enumerate(four_lanes):
        #     color = COLOR_LIST[i]
        #     lane_x, lane_y = [], []
        #     min_y = img_h

        #     points = np.array([lane_i], dtype=np.int32) # (1, 10, 2)
        #     if points.shape[1] > 2:
        #         cv2.fillPoly(frame, points, color)

        name = names[j]
        cv2.imwrite(os.path.join(output_path, name), frame)
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            fp.write(
                                '%d %d ' % (int(out_j[k, i] * col_sample_w * 1640 / 800) - 1, int(590 - k * 20) - 1)
                            )
                    fp.write('\n')


if __name__ == '__main__':
    main()