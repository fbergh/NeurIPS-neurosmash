class Preprocessor:
    def __init__(self, n_channels, crop_values):
        self.left, self.top, self.right, self.bottom = self._cleanup_crop_values(crop_values)
        self.channels_to_keep = self._get_channels_to_keep(n_channels)

    def preprocess(self, img):
        img = img / 255
        img = self._reduce_channels(img)
        img = self._crop(img)
        return img

    def _cleanup_crop_values(self, crop_values):
        left, top, right, bottom = crop_values
        if bottom == 0: 
            bottom = None
        if right == 0: 
            right = None
        assert not left ==  right and not  top ==  bottom, "Crop values cannot be equal"
        return left, top, right, bottom

    def _get_channels_to_keep(self, n_channels):
        channels_to_keep = [0, 1, 2]
        if n_channels == 1:
            # Keep red channel
            channels_to_keep = [0]
        elif n_channels == 2:
            # Keep red and blue channel
            channels_to_keep = [0, 2]
        return channels_to_keep

    def _reduce_channels(self, img):
        return img[:, :, self.channels_to_keep]

    def _crop(self, img):
        img = img[self.top:-self.bottom] if self.bottom else img[self.top:]
        img = img[:, self.left:-self.right] if self.right else img[:, self.left:]
        return img