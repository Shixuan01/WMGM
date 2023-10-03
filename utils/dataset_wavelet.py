from pytorch_wavelets import DWTForward, DWTInverse

import torch
def wavelet_transform(img_tensors: torch.Tensor, channels, size, device):
    xfm = DWTForward(J=1, mode='periodization', wave='db1').to(device)

    Yls = []
    Yhs = []
    for channel in range(3):
        channel_tensor = img_tensors[:, channel, :, :].unsqueeze(1)
        Yl, Yh = xfm(channel_tensor)
        Yls.append(Yl)
        Yhs.append(Yh)

    LL = torch.cat(Yls, dim=1)
    LH = torch.cat([Yh[0][:, :, 0:1, :, :] for Yh in Yhs], dim=1).squeeze(dim=2)
    HL = torch.cat([Yh[0][:, :, 1:2, :, :] for Yh in Yhs], dim=1).squeeze(dim=2)
    HH = torch.cat([Yh[0][:, :, 2:3, :, :] for Yh in Yhs], dim=1).squeeze(dim=2)

    return LL, torch.cat([LH, HL, HH], dim=1)


def wavelet_inverse( combined_tensors: torch.Tensor,channels, image_size, device):
    ifm = DWTInverse(mode='periodization', wave='db1').to(device)
    image_size = image_size//2
    LL = combined_tensors[:, :, :image_size, :image_size]
    LH = combined_tensors[:, :, :image_size, image_size:]
    HL = combined_tensors[:, :, image_size:, :image_size]
    HH = combined_tensors[:, :, image_size:, image_size:]

    Yh = [torch.stack([LH, HL, HH], dim=2)]
    reconstructed_images = ifm((LL, Yh))


    return reconstructed_images

# if __name__ == '__main__':
#     main()