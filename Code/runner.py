import time
import torch
import torch.nn.functional as F

def train_model(model, train_loader, criterion, optimizer, device, measure_accuracy=False):
    model.train()
    running_loss = 0.0
    if measure_accuracy:
        total_predictions = 0.0
        correct_predictions = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.long().to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
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
