import torch
from torch.nn import functional as F

class PGD:
    """

    """

    @staticmethod
    def attack(image, label, device, model, epsilon=0.3, steps=50, step_size = 2/255, no_kl=True):
        model.zero_grad()
        # random start
        perturbation = torch.zeros_like(image).uniform_(-epsilon, epsilon)
        perturbation.requires_grad = True
        perturbed_image = image + perturbation
        perturbed_image.to(device)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image.detach().requires_grad_(True)
        for _ in range(steps):
            output = model(perturbed_image, no_kl)
            loss = F.nll_loss(output, label)
            #loss.backward()
            grad = torch.autograd.grad(loss, perturbed_image, retain_graph=False, create_graph=False)[0]
            sign_data_grad = grad.sign()
            perturbed_image = perturbed_image + sign_data_grad * step_size
            # clamp pertubation only to (-epsilon, epsilon)
            perturbation = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
            # clamp the whole image to (0, 1)
            perturbed_image = torch.clamp(image + perturbation, min=0, max=1)
            #perturbed_image.grad.zero_()
        return perturbed_image