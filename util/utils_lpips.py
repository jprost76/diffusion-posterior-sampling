import torch
import lpips

class LPIPS_np():

    def __init__(self):
        self.loss = lpips.LPIPS(net='alex')

    def __call__(self, img1, img2):
        """
            img1, img2: np.array HWC, in [0, 1]
        """
        t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).type(torch.float)
        t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).type(torch.float)
        # lpips input must be in range [-1, 1]
        t1 = (t1 - 0.5) * 2
        t2 = (t2 - 0.5) * 2
        lpips_loss = self.loss(t1, t2)
        return lpips_loss.item()