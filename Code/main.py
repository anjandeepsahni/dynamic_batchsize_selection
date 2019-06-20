import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from runner import train_model, val_model
import torchvision.transforms as transforms
from checkpoint import load_checkpoint, save_checkpoint
from utils import build_cifar10, parse_args, set_random_seed, init_weights, get_current_lr


CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":

    args = parse_args()

    # set the random seed
    set_random_seed(args.random_seed)

    # transformation for images.
    transform = transforms.Compose(
            [transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print('*'*30)
    print('Setting up dataset...')
    print('*'*30)
    _, _, train_loader, test_loader = build_cifar10(root=args.data_path,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    print('*'*30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device:', device)
    print('*'*30)

    print('Model Architecture')
    model = torchvision.models.alexnet(pretrained=False)
    model = model.to(device)
    model_params = sum(p.size()[0] * p.size()[1] if len(p.size()) > 1 else p.size()[0] for p in model.parameters())
    print(model)
    print('Total model parameters:', model_params)
    print('*'*30)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.01, verbose=True)

    n_epochs = 200
    if args.resume:
        start_epoch, model, optimizer, scheduler = load_checkpoint(model_path=args.model_path,
                                                        ckpt_name=args.ckpt, device=device,
                                                        model=model, optimizer=optimizer,
                                                        scheduler=scheduler)
        start_epoch -= 1
        print('Resumed checkpoint {} from {}. Starting at epoch {}.'.format(args.ckpt, args.model_path, start_epoch+1))
        print('Current learning rate: {}'.format(get_current_lr(optimizer)))
        print('*'*30)
    else:
        start_epoch = 0
        # model = init_weights(model)

    for epoch in range(start_epoch, n_epochs):
        print('Epoch: %d/%d' % (epoch+1,n_epochs))
        train_loss = train_model(model, train_loader, criterion, optimizer, device, measure_accuracy=True)
        val_loss, val_acc = val_model(model, test_loader, criterion, device)
        # Checkpoint the model after each epoch.
        save_checkpoint(epoch+1, args.model_path, model=model, optimizer=optimizer, val_metric=val_acc)
        scheduler.step(val_loss)
        print('='*20)
