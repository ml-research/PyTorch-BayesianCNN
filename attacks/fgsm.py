import torch

class FGSM:
    """
    Fast Gradient Sign Attack https://arxiv.org/abs/1412.6572
    Code from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    """

    @staticmethod
    def attack(image, data_grad, epsilon=0.3):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image