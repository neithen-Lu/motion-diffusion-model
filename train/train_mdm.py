# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import logging
import torch

def main():
    os.environ["OMP_NUM_THREADS"] = "10"
    torch.set_num_threads(10)
    args = train_args()
    fixseed(args.seed)
    if args.dataset in ['humanml','kit']:
        args.num_frames = 196
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    logging.basicConfig(filename= f'{args.save_dir}/train.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    logging.info('start logging')


    dist_util.setup_dist(args.device)

    logging.info("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    logging.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    logging.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    logging.info("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
