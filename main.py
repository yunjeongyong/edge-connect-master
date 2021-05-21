import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect


def main(mode=None, **kwargs):    # 1: train, 2: test, 3: eval
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # config = load_config(mode)
    config = load_config2(mode, **kwargs)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = EdgeConnect(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/indoor', help='model checkpoints path (default: ./checkpoints)')
    # parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    parser.add_argument('--model', type=int, default=2, help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, default='./examples/indoor/images', help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, default='./examples/indoor/masks', help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, default='./examples/indoor/results', help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3  # args.model이 NULL이면 3 그렇지 않으면 입력받은 값
        config.INPUT_SIZE = 0

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


def load_config2(mode=None, **parameter_dict):
    r"""loads model config

        Args:
            mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
        """

    print(parameter_dict)
    parameter_dict.path = parameter_dict['checkpoints']
    config_path = os.path.join(parameter_dict.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(parameter_dict.path):
        os.makedirs(parameter_dict.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if parameter_dict.model:
            config.MODEL = parameter_dict.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = parameter_dict.model if parameter_dict.model is not None else 3  # kwargs.model이 NULL이면 3 그렇지 않으면 입력받은 값
        config.INPUT_SIZE = 0

        if parameter_dict.input is not None:
            config.TEST_FLIST = parameter_dict.input

        if parameter_dict.mask is not None:
            config.TEST_MASK_FLIST = parameter_dict.mask

        if parameter_dict.edge is not None:
            config.TEST_EDGE_FLIST = parameter_dict.edge

        if parameter_dict.output is not None:
            config.RESULTS = parameter_dict.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = parameter_dict.model if parameter_dict.model is not None else 3

    return config


if __name__ == "__main__":
    main()
