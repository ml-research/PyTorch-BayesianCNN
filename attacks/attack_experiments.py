from __future__ import print_function

import os
import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

import data
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
#whitebox attacks
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.attacks.evasion import HighConfidenceLowUncertainty
from art.attacks.evasion import ShadowAttack
#blackbox attacks
from art.attacks.evasion import ZooAttack
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import BoundaryAttack
from art.attacks.evasion import ThresholdAttack
from art.attacks.evasion import SimBA
from art.attacks.evasion import SquareAttack
from art.attacks.evasion import SpatialTransformation

from attacks.pytorch_bayesian import PyTorchBaysianClassifier
from main_bayesian import train_model, getModel, validate_model

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cifar_class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def evaluate_model(net, testloader, classes, device, no_kl=False):
    net.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs, no_kl)
            _, predictions = torch.max(outputs, 1)
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

def test(classifier, attack, test_loader, epsilon=0.3, batch_size=256, no_kl=True):
    # Accuracy counter
    correct = 0
    adv_examples = []

    for x, y in test_loader:
        x_test = x.numpy()
        y_test = y.numpy()

        # Forward pass the data through the model
        output = classifier.predict(x_test, no_kl=no_kl)
        init_pred = output.argmax(1)  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        wrong_indices = np.where(y_test != init_pred)
        if (len(wrong_indices) != len(y_test)):
            x_test = np.delete(x_test, wrong_indices, axis=0)
            y_test = np.delete(y_test, wrong_indices, axis=0)

            perturbed_images = attack.generate(x=x_test)

            # Re-classify the perturbed image
            output = classifier.predict(perturbed_images, no_kl=no_kl)

            # Check for success
            final_pred = output.argmax(1)  # get the index of the max log-probability
            correct_indices = np.where(y_test == final_pred)
            correct += len(correct_indices[0])
            # Save some adv examples for visualization later
            if (len(correct_indices[0]) < len(y_test)):
                if len(adv_examples) < 5:
                    perturbed_images = np.delete(perturbed_images, correct_indices, axis=0)
                    y_test = np.delete(y_test, correct_indices, axis=0)
                    final_pred = np.delete(final_pred, correct_indices, axis=0)
                    adv_ex = perturbed_images.squeeze()
                    adv_examples.append((y_test, final_pred, adv_ex))

    # Calculate final accuracy for this epsilon
    n = len(test_loader) * batch_size
    final_acc = correct / float(n)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def run(dataset, net_type, activation_type):
    import matplotlib.pyplot as plt
    # this code is largely adopted from main_bayesian.py

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    # activation_type = cfg.activation_type
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

    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

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
    classes = [str(x) for x in range(0, 10)]  # for MNIST
    # evaluate accuracy for benign images
    no_kl = True
    evaluate_model(net, test_loader, classes, device, no_kl=no_kl)

    min_pixel_value = 0
    max_pixel_value = 255
    classifier = PyTorchBaysianClassifier(
        model=net,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=metrics.ELBO(len(trainset)).to(device),
        optimizer=Adam(net.parameters(), lr=lr_start),
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    attack_classifier = PyTorchBaysianClassifier(
        model=net,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=CrossEntropyLoss(),  # metrics.ELBO(len(trainset)).to(device),
        optimizer=Adam(net.parameters(), lr=lr_start),
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    accuracies = []
    examples = []

    # Run fgsm attack for each epsilon
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    eps_step = 0.01
    for eps in epsilons:
        attack = FastGradientMethod(estimator=attack_classifier, eps=eps)
        # attack = ProjectedGradientDescentPyTorch(estimator=attack_classifier, eps=eps, eps_step=eps_step, verbose=False)
        # attack = BoundaryAttack(estimator=attack_classifier, epsilon=eps, targeted=False)
        # attack = ZooAttack(classifier=attack_classifier, confidence=eps)
        # attack = SimBA(classifier=attack_classifier, epsilon=eps)
        acc, ex = test(classifier, attack, test_loader, eps, batch_size)
        accuracies.append(acc)
        examples.append(ex)

    # attack = ThresholdAttack(classifier=attack_classifier)
    # attack = HopSkipJump(classifier=attack_classifier)
    # acc, ex = test(classifier, attack, test_loader, batch_size=batch_size)
    # accuracies.append(acc)
    # examples.append(ex)

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

            if dataset == "MNIST":
                plt.title("{} -> {}".format(orig[0], adv[0]))
                plt.imshow(ex[0], cmap="gray")
            else:
                plt.title("{} -> {}".format(cifar_class[orig[0]], cifar_class[adv[0]]))
                plt.imshow(ex)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = 'lenet'
    dataset = 'MNIST'
    activation_type = 'softplus'

    run(dataset, model, activation_type)
