import torch
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import CONFIG
from model import Model
from tqdm import tqdm
# from sklearn.metrics import classification_report


def plot_point_cloud(rows, cols, height, num):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rows, cols, height)
    major_ticks = np.arange(0, 40, 10)
    minor_ticks = np.arange(0, 40, 5)
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_zticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)

    ax.grid(which='both')
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.001)
    ax.grid(True)

    # plt.show()
    plt.savefig('point_cloud_'+ str(num) +'.png')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
    def __init__(self, dataset, labels, transform=None):
        self.data = dataset
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        transformed_dataset = self.data[idx]
        if self.transform:
            transformed_dataset = self.transform(self.data[idx])
            transformed_dataset = torch.unsqueeze(transformed_dataset, 0)
            transformed_dataset = transformed_dataset.to(torch.float)
        return transformed_dataset, torch.tensor(label)
    
def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the device:
    model.to(device)
    # Loop over the epochs:
    for epoch in tqdm(range(num_epochs)):
        # Set the model to training mode:
        model.train()
        # Loop over the training data:
        for x, y in train_loader:
            # Move the data to the device:
            x, y = x.to(device), y.to(device)
            # Zero the gradients:
            optimizer.zero_grad()
            # Forward pass:
            y_hat = model(x)
            # Compute the loss:
            loss = criterion(y_hat, y)
            # Backward pass:
            loss.backward()
            # Update the parameters:
            optimizer.step()
        # Set the model to evaluation mode:
        model.eval()
        # Compute the accuracy on the test data:
        accuracy = compute_accuracy(model, test_loader, device)
        # Print the results:
        if epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch + 1} | Test Accuracy: {accuracy:.2f}")

def compute_accuracy(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """
    Compute the accuracy of a model on some data.
    Arguments:
        model (torch.nn.Module): The model to compute the accuracy of.
        data_loader (DataLoader): The data loader to use.
        device (torch.device): The device to use for training.
    Returns:
        accuracy (float): The accuracy of the model on the data.
    """
    # Set the model to evaluation mode:
    model.eval()
    # Initialize the number of correct predictions:
    num_correct = 0
    y_pred_list = []
    y_list = []

    # Loop over the data:
    for x, y in data_loader:
        # Move the data to the device:
        x, y = x.to(device), y.to(device)
        # Forward pass:
        y_hat = model(x)
        # Compute the predictions:
        predictions = torch.argmax(y_hat, dim=1)
        # Update the number of correct predictions:
        num_correct += torch.sum(predictions == y).item()
        y_pred_list = y_pred_list + list(predictions.cpu().detach().numpy())
        y_list = y_list + list(y.cpu().detach().numpy())
    # Compute the accuracy:
    accuracy = num_correct / len(data_loader.dataset)
    # print(classification_report(y_list, y_pred_list))
    # Return the accuracy
    return accuracy

def main(split_ratio, num_epochs):
    np.random.seed(0)
    print(split_ratio)
    print(num_epochs)
    N = 100
    M = 40
    input = np.zeros((N, M, M, M))
    test_split_num = 5*M*M*M//100
    # total points = 40*40*40: 5% = 3200
    # need to generate 3200 random points between 0 to 40, exclusive

    for i in range(N):
        occupied_cells = np.random.randint(0, 40, size=(3200, 3))
        input[i, occupied_cells] = 1
        occ_list = list(map(tuple, occupied_cells))
        
        oset = set(occ_list)
        while len(oset) < test_split_num:
            diff = test_split_num - len(oset)
            add_occ_cells = np.random.randint(0, 40, size=(diff, 3))
            add_occ_list = list(map(tuple, add_occ_cells))
            oset.update(add_occ_list)
            input[i, add_occ_cells] = 1
        # occ_cells_array = np.array(list(map(list, oset)))
        # plot_point_cloud(list(occ_cells_array[:, 0]), list(occ_cells_array[:, 1]), list(occ_cells_array[:, 2]), i)
    
    K = 5
    labels = np.zeros((N))
    for i in range(K):
        labels[20*i:(20*i)+20] = i
    labels = labels.astype(int)
    dataset = CustomDataset(input, labels, CONFIG.transforms)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_ratio, 1-split_ratio])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=True)

    # Create the model:
    model = Model(num_classes=K)
    # Create the optimizer:
    optimizer = CONFIG.optimizer_factory(model)
    # Create the loss function:
    criterion = torch.nn.CrossEntropyLoss()
    # Train the model:
    train(
        model,
        train_loader,
        val_loader,
        num_epochs,
        optimizer=optimizer,
        criterion=criterion,
    )

if __name__ == "__main__":
    args = parse_args()
    split_ratio = args.split_ratio
    num_epochs = args.epochs
    main(split_ratio, num_epochs)


