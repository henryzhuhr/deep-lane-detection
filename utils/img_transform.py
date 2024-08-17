import cv2
import numpy as np
import torch


class NormalizeValue:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


def tensor2cvmat(image: torch.Tensor) -> cv2.Mat:
    """
    # Convert `torch.Tensor` to `cv2.Mat`

    ## Args:
    -  `image` (`torch.Tensor`): shape must be `[C,H,W]` or `[1,C,H,W]`, value range from `[0, 1]`

    ## Returns:
    - `img` (`cv2.Mat`): shape is `[H,W,C]`, value range from `[0, 255]`
    """
    if (image.dim() == 3):
        pass
    elif (image.dim() == 4):
        if (image.shape[0] == 1):
            image = image.squeeze(0)
        else:
            raise ValueError("tensor's batch size must be 1")
    else:
        raise ValueError("tensor dim must be 3 or 4")
    img: cv2.Mat = image.data.detach().cpu().numpy()
    img = img.transpose(1, 2, 0) # [C,H,W] -> [H,W,C]
    img = img * np.array(NormalizeValue.std) + np.array(NormalizeValue.mean)
    img = img * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def cvmat2tensor(img: cv2.Mat):
    """
    # Convert `cv2.Mat` to `torch.Tensor`

    ## Args:
    -  `img` (`cv2.Mat`): shape must be `[H,W,C]`, value range from `[0, 255]`

    ## Returns:
    - `image` (`torch.Tensor`): shape is `[1,C,H,W]`, value range from `[0, 1]`
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    img = (img - np.array(NormalizeValue.mean)) / np.array(NormalizeValue.std)
    image = torch.from_numpy(img)
    image = image.permute(2, 0, 1) # [H,W,C] -> [C,H,W]
    image = image.unsqueeze(0)     # [C,H,W] -> [1,C,H,W]
    return image
