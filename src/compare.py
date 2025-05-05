import os
import torch
import pickle
import numpy as np
import random
import argparse
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helpers import cluster_acc, plot_confusion_matrix
from PytorchUtils import Seq_data, Net_linear, Net_linear_improved, IID_loss
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd

# הגדרת שימוש ב-MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Random Seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def weights_init(m):
    """ Kaiming initialization of the weights """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def eval_training(net, training_set, x_test, y_test, l=1.0, _lr=0.0001, k=6, epochs=10):
    """ Train and evaluate a given network """
    batch_size = 512
    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(net.parameters(), lr=_lr)

    train_losses, test_accuracies , all_predictions, = [], [],[]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} running...")
        net.train()
        epoch_loss = 0

        for sample_batched in dataloader:
            sample = sample_batched['true'].view(-1, 1, 2 ** k, 2 ** k).to(device)
            modified_sample = sample_batched['modified'].view(-1, 1, 2 ** k, 2 ** k).to(device)

            optimizer.zero_grad()
            z1 = net(sample)
            z2 = net(modified_sample)

            loss = IID_loss(z1, z2, lamb=l)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(dataloader))

        # Testing Process
        net.eval()
        predicted, y_true = [], []

        for i in range(x_test.shape[0]):
            sample = torch.from_numpy(x_test[i]).to(device)
            label = y_test[i]
            sample = sample.view(1, 1, 2 ** k, 2 ** k).to(device)
            output = net(sample)

            top_n, top_i = output.topk(1)
            predicted.append(top_i[0].item())
            y_true.append(label)

        predicted = np.array(predicted)
        y_true = np.array(y_true)

        ind, acc = cluster_acc(y_true, predicted)
        d = {}
        for i, j in ind:
            d[i] = j

        for i in range(x_test.shape[0]):  # we do this for each sample or sample batch
            predicted[i] = d[predicted[i]]
        test_accuracies.append(acc)
        all_predictions.append(predicted)


    return train_losses, test_accuracies, all_predictions


def evaluate_models(data_dir, num_runs=5, epochs=10):
    """ Evaluate both models over multiple runs and epochs. """
    filename = os.path.join(data_dir, 'mimics.p')
    x_train, x_test, y_test = pickle.load(open(filename, 'rb'))
    num_classes = len(set(y_test))

    # סקלינג הנתונים בנפרד
    scaler = StandardScaler()
    scaler.fit(x_test)

    x_train_1 = scaler.transform(x_train[:, 0, :])
    x_train_2 = scaler.transform(x_train[:, 1, :])
    x_test = scaler.transform(x_test)

    x_train[:, 0, :] = x_train_1
    x_train[:, 1, :] = x_train_2

    training_set = Seq_data(x_train)
    num_heads_list = [4, 8, 16]

    results = []
    train_losses_dict = {}
    test_accuracies_dict = {}

    predictions = []
    accuracies = []
    all_train_losses = []
    all_test_accuracies = []

    print("\n🔹 Testing Original Model 🔹")
    for i in range(num_runs):
        l = 2.8
        _lr = 8.e-5

        net_original = Net_linear(4 ** 6, num_classes).to(device)
        net_original.apply(weights_init)

        train_losses, test_accuracies, predicted = eval_training(net_original, training_set, x_test, y_test, l=l,
                                                                 _lr=_lr, k=6, epochs=epochs)

        predictions.append(predicted[-1])  # שמירת התחזיות מה-epoch האחרון
        accuracies.append(test_accuracies[-1])
        all_train_losses.append(train_losses)
        all_test_accuracies.append(test_accuracies)

    predictions = np.array(predictions)
    mode, counts = stats.mode(predictions, axis=0)

    w = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(y_test.shape[0]):
        w[y_test[i], int(mode[i])] += 1

    final_accuracy = np.sum(np.diag(w) / np.sum(w))
    print("Confusion Matrix:")
    print(w)
    print("Final Accuracy:", final_accuracy)

    results.append(["Original", None, final_accuracy])
    train_losses_dict["Original"] = np.mean(all_train_losses, axis=0)
    test_accuracies_dict["Original"] = np.mean(all_test_accuracies, axis=0)

    for num_heads in num_heads_list:
        print(f"\n🔹 Testing Improved Model with num_heads={num_heads} 🔹")
        predictions = []
        accuracies = []
        all_train_losses = []
        all_test_accuracies = []

        for i in range(num_runs):
            net_improved = Net_linear_improved(4 ** 6, num_classes, num_heads=num_heads).to(device)
            net_improved.apply(weights_init)

            train_losses, test_accuracies, predicted = eval_training(net_improved, training_set, x_test, y_test,
                                                                     epochs=epochs)

            predictions.append(predicted[-1])
            accuracies.append(test_accuracies[-1])
            all_train_losses.append(train_losses)
            all_test_accuracies.append(test_accuracies)

        predictions = np.array(predictions)
        mode, counts = stats.mode(predictions, axis=0)

        w = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i in range(y_test.shape[0]):
            w[y_test[i], int(mode[i])] += 1

        final_accuracy = np.sum(np.diag(w) / np.sum(w))
        print("Confusion Matrix:")
        print(w)
        print("Final Accuracy:", final_accuracy)

        results.append(["Improved", num_heads, final_accuracy])
        train_losses_dict[f"Improved-{num_heads} Heads"] = np.mean(all_train_losses, axis=0)
        test_accuracies_dict[f"Improved-{num_heads} Heads"] = np.mean(all_test_accuracies, axis=0)

    # שמירת התוצאות לטבלה
    df_results = pd.DataFrame(results, columns=["Model Type", "Num Heads", "Final Test Accuracy"])
    df_results.to_csv("model_results_Bacteria.csv", index=False)
    print("\n🔹 Results saved to model_results.csv 🔹")

# יצירת גרף Loss לכל המודלים
    plt.figure(figsize=(8, 6))
    for model_name, losses in train_losses_dict.items():
        plt.plot(range(1, epochs + 1), losses, marker='o', label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_comparison_Bacteria.png")
    plt.show()

    # יצירת גרף Accuracy לכל המודלים
    plt.figure(figsize=(8, 6))
    for model_name, accuracies in test_accuracies_dict.items():
        plt.plot(range(1, epochs + 1), accuracies, marker='s', label=model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_accuracy_comparison_Bacteria.png")
    plt.show()

if __name__ == '__main__':
    evaluate_models('../data/Bacteria', num_runs=5, epochs=10)