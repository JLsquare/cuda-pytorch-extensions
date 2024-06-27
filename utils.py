import os
import time
import torch
import matplotlib.pyplot as plt

def train_model(
        model: torch.nn.Module,
        trainLoader: torch.utils.data.DataLoader,
        validationLoader: torch.utils.data.DataLoader,
        train_set: torch.utils.data.Dataset,
        validation_set: torch.utils.data.Dataset,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        model_path: str,
        save_every: int,
        epochs: int = 8
) -> tuple:
    train_loss, val_loss = [], []
    accuracy_total_train, accuracy_total_val = [], []
    start_time = time.time()
    last_time = start_time

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        model.train()
        total = 0

        for idx, (image, label) in enumerate(trainLoader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()

            pred = model(image)
            loss = criterion(pred, label)

            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total += 1

        accuracy_train = total / len(train_set)
        accuracy_total_train.append(accuracy_train)
        total_train_loss = total_train_loss / (idx + 1)
        train_loss.append(total_train_loss)

        model.eval()
        total = 0

        with torch.no_grad():
            for idx, (image, label) in enumerate(validationLoader):
                image, label = image.to(device), label.to(device)

                pred = model(image)
                loss = criterion(pred, label)

                total_val_loss += loss.item()
                pred = torch.nn.functional.softmax(pred, dim=1)
                for i, p in enumerate(pred):
                    if label[i] == torch.max(p.data, 0)[1]:
                        total += 1

        accuracy_val = total / len(validation_set)
        accuracy_total_val.append(accuracy_val)
        total_val_loss = total_val_loss / (idx + 1)
        val_loss.append(total_val_loss)

        print("Epoch: {}/{}  ".format(epoch + 1, epochs),
              "Train loss: {:.4f}  ".format(total_train_loss),
              "Valid loss: {:.4f}  ".format(total_val_loss),
              "Train acc: {:.4f}  ".format(accuracy_train),
              "Valid acc: {:.4f}  ".format(accuracy_val),
              "Elapsed time: {:.4f} sec  ".format(time.time() - start_time),
              "Time left: {:.4f} sec".format((time.time() - last_time) * (epochs - epoch - 1)))

        last_time = time.time()

        if epoch != 0 and save_every != 0 and (epoch + 1) % save_every == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), model_path + "epoch_" + str(epoch) + ".pt")
            print("Model saved!")

    total_time = time.time() - start_time

    return train_loss, val_loss, accuracy_total_train, accuracy_total_val, total_time

def plot_loss(train_loss: list, val_loss: list):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.show()

def plot_accuracy(accuracy_total_train: list, accuracy_total_val: list):
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_total_train, label='Training accuracy')
    plt.plot(accuracy_total_val, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.show()

def test_accuracy(model: torch.nn.Module, validationLoader: torch.utils.data.DataLoader, validation_set: torch.utils.data.Dataset, device: str) -> float:
    model.eval()
    total = 0
    for idx, (image, label) in enumerate(validationLoader):
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            pred = model(image)
            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

    accuracy = total / len(validation_set)
    return accuracy

def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))