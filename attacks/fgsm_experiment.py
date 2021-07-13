from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from attacks.fgsm import FGSM
from main_bayesian import train_model, getModel, validate_model

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model(net, testloader, classes, device):
    net.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs[0], 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))

def load_checkpoint(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))

def test(model, device, test_loader, epsilon=0.3, batch_size=256):
    #this code is taken from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for images, targets in test_loader:
        for data, target in zip(images, targets):
            data = torch.unsqueeze(data, 0)
            target = torch.unsqueeze(target, 0)

            # Send the data and label to the device
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)[0]
            init_pred = output.max(1)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.tolist() != target.tolist(): # but this now moves on over a bunch of images?
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = FGSM.attack(data, data_grad, epsilon)

            # Re-classify the perturbed image
            output = model(perturbed_data)[0]

            # Check for success
            final_pred = output.max(1)[1] # get the index of the max log-probability
            if final_pred.tolist() == target.tolist():
                correct += 1 # this case is for unsuccessfull attacks
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.tolist(), final_pred.tolist(), adv_ex) )

    # Calculate final accuracy for this epsilon
    n = len(test_loader) * batch_size
    final_acc = correct/float(n)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def run(dataset, net_type):
    import matplotlib.pyplot as plt
    #this code is largely adopted from main_bayesian.py

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_name = f'../checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if os.path.exists(ckpt_name):
        load_checkpoint(net, ckpt_name)
    else:
        criterion = metrics.ELBO(len(trainset)).to(device)
        optimizer = Adam(net.parameters(), lr=lr_start)
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
        valid_loss_max = np.Inf

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                                          beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
            valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type,
                                                   epoch=epoch, num_epochs=n_epochs)
            lr_sched.step(valid_loss)

            print(
                'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

            # save model if validation accuracy has increased
            if valid_loss <= valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_max, valid_loss))
                torch.save(net.state_dict(), ckpt_name)
                valid_loss_max = valid_loss
    classes = [str(x) for x in range(0, 10)] # for MNIST
    # evaluate accuracy for benign images
    evaluate_model(net, test_loader, classes, device)

    accuracies = []
    examples = []

    # Run fgsm attack for each epsilon
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    for eps in epsilons:
        acc, ex = test(net, device, test_loader, eps, batch_size)
        accuracies.append(acc)
        examples.append(ex)

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig[0], adv[0]))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = 'lenet'
    dataset ='MNIST'

    run(dataset, model)