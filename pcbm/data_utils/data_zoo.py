from torchvision import datasets
import torch
import os


def get_dataset(args, preprocess=None):
    if args.dataset == "Waterbirds":
        from .waterbirds import load_waterbirds_data
        from .constants import WATERBIRDS_DATA_DIR, WATERBIRDS_METADATA
        num_classes = 2
        train_loader = load_waterbirds_data(is_training=True, batch_size=args.batch_size, metadata_path=WATERBIRDS_METADATA,
                        image_dir=WATERBIRDS_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)

        test_loader = load_waterbirds_data(is_training=False, batch_size=args.batch_size, metadata_path=WATERBIRDS_METADATA,
                        image_dir=WATERBIRDS_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)
        
        idx_to_class = {0: "Landbird", 1: "Waterbird"}
        classes = list(idx_to_class.values())
    
    elif args.dataset == "MetaShift":
        from .metashift import load_metashift_data
        from .constants import METASHIFT_DATA_DIR, METASHIFT_METADATA
        num_classes = 2
        train_loader = load_metashift_data(is_training=True, batch_size=args.batch_size, metadata_path=METASHIFT_METADATA,
                        image_dir=METASHIFT_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)

        test_loader = load_metashift_data(is_training=False, batch_size=args.batch_size, metadata_path=METASHIFT_METADATA,
                        image_dir=METASHIFT_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)
        
        idx_to_class = {0: "Cat", 1: "Dog"}
        classes = list(idx_to_class.values())

    elif args.dataset == "CelebA":
        from .celeba import load_celeba_data
        from .constants import CELEBA_DATA_DIR, CELEBA_METADATA
        num_classes = 2
        train_loader = load_celeba_data(is_training=True, batch_size=args.batch_size, metadata_path=CELEBA_METADATA,
                        image_dir=CELEBA_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)

        test_loader = load_celeba_data(is_training=False, batch_size=args.batch_size, metadata_path=CELEBA_METADATA,
                        image_dir=CELEBA_DATA_DIR, resol=224, preprocess=preprocess, n_classes=num_classes, resampling=False)
        
        idx_to_class = {0: "Dark Hair", 1: "Blonde Hair"}
        classes = list(idx_to_class.values())
    else:
        raise ValueError(args.dataset)

    return train_loader, test_loader, idx_to_class, classes

