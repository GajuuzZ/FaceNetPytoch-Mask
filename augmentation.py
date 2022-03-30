from imgaug import augmenters as iaa


def face_augment_pipe():
    sometimes = lambda a: iaa.Sometimes(0.5, a)
    aug_pipe = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                rotate=(-10, 10),
                scale=(0.9, 1.1),
                mode=['constant', 'edge']
            )),
            iaa.OneOf([
                iaa.Noop(),
                iaa.GaussianBlur((0, 3)),
                iaa.MedianBlur(k=(1, 7)),
            ]),
            iaa.SomeOf((0, None),
                       [
                           iaa.Add((-20, 20)),  # change brightness of images.
                           iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                           iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.8)),
                           iaa.Noop()
                       ],
                       random_order=True
                       )
        ],
        random_order=True,
    )
    return aug_pipe
