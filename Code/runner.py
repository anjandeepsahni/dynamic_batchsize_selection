import time
import torch
from utils import set_current_lr, get_current_lr
import torch.nn.functional as F

GRADIENT_CLIP = 5

def opti_batch_processing(model, criterion, data, target):
    base_bs = len(data)
    assert(base_bs >= 256)
    bs_list = [base_bs, base_bs>>1, base_bs>>2]
    o_list, l_list, l_val_list = [], [], []
    for bs in bs_list:
        splits = int(base_bs/bs)
        l_sublist, o_sublist = [], []
        curr_loss = 0
        for split in range(splits):
            start = bs*split
            end = bs*split + bs
            d = data[start:end]
            t = target[start:end]
            o = model(d)
            l = criterion(o, t)
            curr_loss += l.item()
            l_sublist.append(l)
            o_sublist.append(o)
        l_list.append(l_sublist)
        l_val_list.append(curr_loss/len(l_sublist))
        o_list.append(torch.cat(o_sublist))
    best_batch_idx = l_val_list.index(min(l_val_list))
    best_loss = l_val_list[best_batch_idx]
    losses = l_list[best_batch_idx]
    outputs = o_list[best_batch_idx]
    lr_factor = len(losses)
    return outputs, best_loss, losses, lr_factor

def train_model(model, train_loader, criterion, optimizer, device, measure_accuracy=False, opti_batch=False):
    model.train()
    running_loss = 0.0
    if measure_accuracy:
        total_predictions = 0.0
        correct_predictions = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.long().to(device)
        if opti_batch:
            outputs, best_loss, losses, lr_factor = opti_batch_processing(model, criterion, data, target)
            running_loss += best_loss
            old_lr = get_current_lr(optimizer)
            new_lr = old_lr / lr_factor
            optimizer = set_current_lr(optimizer, new_lr)
            for l in losses:
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)   # for exploding nan grad.
                optimizer.step()
            optimizer = set_current_lr(optimizer, old_lr)
        else:
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)   # for exploding nan grad.
            optimizer.step()
        if measure_accuracy:
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            predicted = predicted.view(-1)
            total_predictions += target.size(0)
            correct_predictions += torch.sum(torch.eq(predicted, target)).item()
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0 if measure_accuracy else -1
    print('\nTraining Loss: %5.4f Training Accuracy: %5.3f Time: %d s' % \
            (running_loss, acc, end_time - start_time))
    return running_loss

def val_model(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.long().to(device)
            outputs = model(data)
            _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            predicted = predicted.view(-1)
            total_predictions += target.size(0)
            correct_predictions += torch.sum(torch.eq(predicted, target)).item()
            loss = criterion(outputs, target)
            running_loss += loss.item()
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        running_loss /= len(val_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('\nValidation Loss: %5.4f Validation Accuracy: %5.3f Time: %d s' % \
                (running_loss, acc, end_time - start_time))
    return running_loss, acc
