from mxnet.gluon.nn.basic_layers import HybridBlock, Block
from mxnet.gluon.data.vision import transforms
import mxnet.symbol as s
import mxnet
from PIL.Image import Image


DEFAULT_CROP_RATIO = 28 / 96


class Crop(HybridBlock):
    def __init__(self, crop_values):
        super(Crop, self).__init__()
        self.left, self.top, self.right, self.bottom = crop_values
        if self.bottom == 0:
            self.bottom = None
        if self.right == 0:
            self.right = None
        assert not self.left == self.right and not self.top == self.bottom, "Crop values cannot be equal"

    def hybrid_forward(self, F, x, *args, **kwargs):
        if isinstance(x, Image):
            raise Exception("Cannot crop PIL image")
        if self.bottom is None:
            x = x[self.top:]
        else:
            x = x[self.top:-self.bottom]
        if self.right is None:
            x = x[:, self.left:]
        else:
            x = x[self.left:-self.right]
        return x


class ReduceChannels(HybridBlock):
    def __init__(self, to_keep):
        super(ReduceChannels, self).__init__()
        self.to_keep = to_keep#mxnet.ndarray.array(to_keep)

    def hybrid_forward(self, F, x, *args, **kwargs):
        if isinstance(x, Image):
            raise Exception("Cannot crop PIL image")
        # Use function space F to take to_keep channels on the channel axis
        return F.take(a=x, indices=s.Symbol(self.to_keep), axis=2)


def get_preprocess_transform(img_size, n_channels):
    """
    Normalises and crops image.
    Optionally reduces the number of channels as specified by n_channels
    """
    channels_to_keep = [0, 1, 2]
    if n_channels == 1:
        # Keep red channel
        channels_to_keep = [0]
    elif n_channels == 2:
        # Keep red and blue channel
        channels_to_keep = [0, 2]

    preprocess = transforms.Compose([
        # Convert PIL to tensor and normalise from [0,255] to [0,1]
        transforms.ToTensor(),
        ReduceChannels(channels_to_keep),
        # Crop ratio ensures that varying image sizes don't matter
        Crop((0, int(img_size * DEFAULT_CROP_RATIO), 0, 0))
    ])

    return preprocess
