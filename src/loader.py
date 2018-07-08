import argparse
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import datasets

try:
    from .pytorch_dcgan import train
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from pytorch_dcgan import train


def mybool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Please use one of 'yes', 'y', 'true', 't', '1' or the counterpart.")


def loader(name, dataset, channels):
    """
    Only supports MNIST and CIFAR10 for now.

    :param name: 'mnist' or 'cifar'
    :param dataset: (torchvision.datasets.)MNIST oder CIFAR10
    :param channels: 1 or 3 (for color-depth)
    """

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("-name", help="Name for all outputs (def: %s)" % name, type=str, default=name)
    parser.add_argument("-epochs", help="Amount of epochs (def: 20)", type=int, default=20)
    parser.add_argument("-classes", help="If true, train supervised (def: yes)", type=mybool, default=True)
    parser.add_argument("-lr", help="Starting learning-rate (def: 0.0002)", type=float, default=0.0002)
    parser.add_argument("-decay", help="If true, lr-decay is applied all n epochs (def: yes)",
                        type=mybool, default=True)
    parser.add_argument("-decay-n", help="Every n-th step, decay is applied (def: 5)", type=int, default=5)
    parser.add_argument("-decay-m", help="Every decay step, lr is multiplied by m (1=no decay; def: .3)",
                        type=float, default=.3)
    parser.add_argument("-net-resolution", help="Net-Resolution (def: 128)", type=int, default=100)
    parser.add_argument("-mbd", help="Size of MBD layer (0 for none, def: 8)", type=int, default=8)
    parser.add_argument("-save-nth-net", help="If set, will export the net every n-th epoch (def: 0=do not export)",
                        type=int, default=0)
    parser.add_argument("-clean", help="If yes, will delete old models and images (def: no)",
                        type=mybool, default=False)
    parser.add_argument("-feature-match", help="Weight of feature-match loss (0 = off, def: 10)", type=int, default=10)
    parser.add_argument("-img-size", help="Size of the generated images (def: 64)", type=int, default=64)
    parser.add_argument("-batch-size", help="Size of one mini-batch (def: 128)", type=int, default=128)
    parser.add_argument("-gen-min-quality", help="Minimum of generator loss at the end of each epoch (def: off)", type=float, default=0)
    args = parser.parse_args()
    
    if not args.classes:
        print("Class-GAN disabled.")

    net_resolution = args.net_resolution  # default for MNIST: 128

    img_size = args.img_size
    batch_size = args.batch_size
    channels = channels  # 3 Channels (RGB) / 1 Channel B/W

    # data_loader
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,) * channels, std=(0.5,) * channels)
    ])
    train_loader = DataLoader(
        dataset(
            'data',
            train=True,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=True)

    train(
        name=args.name,
        train_loader=train_loader,
        channels=channels,
        net_resolution=net_resolution,
        train_epoch=args.epochs,
        batch_size=batch_size,
        lr=args.lr,
        mbl_size=args.mbd,
        train_classes=args.classes,
        decay=((args.decay_n, args.decay_m) if args.decay else (1, 1)),
        save_nth_net=args.save_nth_net,
        clean=args.clean,
        feature_match_weight=args.feature_match,
        gen_min_quality=args.gen_min_quality
    )


if __name__ == '__main__':  # default: run cifar
    loader('cifar', datasets.CIFAR10, 3)
