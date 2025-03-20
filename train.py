from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.backends.cudnn as cudnn

from models import Autoencoder, toTensor, var_to_np
from util import get_image_paths, load_images, stack_images
from training_data import get_training_data


def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N', help='Number of epochs to train (default: 100000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Enables CPU training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Logging interval (default: 100)')
    return parser.parse_args()


def init_device_and_seed(args):
    if args.cuda and torch.cuda.is_available():
        print("===> Using GPU for training")
        device = torch.device("cuda:0")
        cudnn.benchmark = True
    else:
        print("===> Using CPU for training")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return device


def load_datasets():
    print("===> Loading datasets")
    images_A = get_image_paths("data/trump")
    images_B = get_image_paths("data/cage")
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0


    images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
    return images_A, images_B

def load_checkpoint(model, checkpoint_path):
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state'])
            start_epoch = checkpoint['epoch']
            print("===> Loaded checkpoint")
        except FileNotFoundError:
            print("Checkpoint file not found, starting from scratch")
            start_epoch = 0
    else:
        start_epoch = 0
        print("Checkpoint folder not found, starting from scratch")
    
    return start_epoch


def train_model(model, device, images_A, images_B, criterion, optimizer_1, optimizer_2, start_epoch, args):
    print("===> Starting training, press 'q' to quit")
    
    for epoch in range(start_epoch, args.epochs):
        batch_size = args.batch_size


        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)


        warped_A, target_A = toTensor(warped_A), toTensor(target_A)
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)


        if args.cuda:
            warped_A, target_A = warped_A.to(device).float(), target_A.to(device).float()
            warped_B, target_B = warped_B.to(device).float(), target_B.to(device).float()


        optimizer_1.zero_grad()
        optimizer_2.zero_grad()


        warped_A = model(warped_A, 'A')
        warped_B = model(warped_B, 'B')


        loss1 = criterion(warped_A, target_A)
        loss2 = criterion(warped_B, target_B)
        total_loss = loss1.item() + loss2.item()


        loss1.backward()
        loss2.backward()


        optimizer_1.step()
        optimizer_2.step()


        print(f"Epoch: {epoch}, LossA: {loss1.item()}, LossB: {loss2.item()}")


        if epoch % args.log_interval == 0:
            save_checkpoint(model, epoch)


        visualize_results(model, target_A, target_B)


        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break


def save_checkpoint(model, epoch):
    checkpoint_path = os.path.join('checkpoint', 'autoencoder.t7')
    if not os.path.isdir('checkpoint'):
        os.makedirs('checkpoint')
    state = {'state': model.state_dict(), 'epoch': epoch}
    torch.save(state, checkpoint_path)
    print("===> Model checkpoint saved")


def visualize_results(model, target_A, target_B):
    test_A_ = target_A[0:14]
    test_B_ = target_B[0:14]
    
    test_A = var_to_np(target_A[0:14])
    test_B = var_to_np(target_B[0:14])

    figure_A = np.stack([test_A, var_to_np(model(test_A_, 'A')), var_to_np(model(test_A_, 'B'))], axis=1)
    figure_B = np.stack([test_B, var_to_np(model(test_B_, 'B')), var_to_np(model(test_B_, 'A'))], axis=1)

    figure = np.concatenate([figure_A, figure_B], axis=0)
    figure = figure.transpose((0, 1, 3, 4, 2))
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = np.clip(figure * 255, 0, 255).astype('uint8')
    cv2.imshow("", figure)

def main():
    
    args = parse_args()

    
    device = init_device_and_seed(args)

    
    images_A, images_B = load_datasets()

    
    model = Autoencoder().to(device)
    criterion = nn.L1Loss()

    
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_A.parameters()}], lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                              {'params': model.decoder_B.parameters()}], lr=5e-5, betas=(0.5, 0.999))

    
    start_epoch = load_checkpoint(model, './checkpoint/autoencoder.t7')

    
    train_model(model, device, images_A, images_B, criterion, optimizer_1, optimizer_2, start_epoch, args)

if __name__ == "__main__":
    main()
