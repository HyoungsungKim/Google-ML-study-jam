from tensorboardX import SummaryWriter
import torch

import layer
import sl_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path):
    model = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def write_histogram(model_path):
    model = load_model(model_path)
    sl = sl_train.SL()
    distribution, _, target = sl.get_distribution(model)

    writer = SummaryWriter(logdir='histogram/accuracy-95')
    frequency_checker = {}
    for dist, tar in zip(distribution, target):
        step = 0
        str_tar = str(tar.item())
        if str_tar in frequency_checker:
            step = frequency_checker[str_tar]
        else:
            frequency_checker[str_tar] = 0

        writer.add_histogram(tag=str_tar, values=dist, global_step=step)        
        frequency_checker[str_tar] += 1


if __name__ == "__main__":
    write_histogram('./model/accuracy95.pth')
    print("Done")
