import os.path
from collections import defaultdict

from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from src.losses import MultiClassDiceLoss
from src.utils import *
from src.datasets import ExcavatorDataset
from src.config import DEVICE, TRANSFORMER
from models.Segmentation import _BaseSegmentationModel

def compute_and_print_class_ratios(train_dataset: Dataset,
                                   val_dataset: Dataset,
                                   class_colors: dict) -> None:
    """
    Compute and print pixel ratios between classes in training and validation datasets.

    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param class_colors: Class colors dictionary.
    """
    class_indices = list(i for i in range(len(class_colors)))

    def compute_class_pixel_counts(dataset: Dataset):
        class_pixel_counts = defaultdict(int)
        total_pixels = 0

        for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
            _, mask, *_ = dataset[idx]  # Assuming dataset returns (image, mask)
            if not mask.ndim == 2:
                if mask.shape[0] != 3:
                    raise RuntimeError(f"Mask should be a 2D tensor of size (H, W) or a 3D tensor of size (C, H, W), got {mask.shape}")
                mask = rgb_to_mask(mask, class_colors)  # Shape: (H, W)

            # Flatten the mask to 1D for counting
            mask_flat = mask.view(-1)

            # Count pixels per class
            for class_idx in torch.unique(mask_flat):
                class_idx = int(class_idx.item())
                pixel_count = torch.sum(mask_flat == class_idx).item()
                class_pixel_counts[class_idx] += pixel_count
                total_pixels += pixel_count

        return class_pixel_counts, total_pixels

    # Compute class pixel counts for training and validation datasets
    train_class_counts, train_total_pixels = compute_class_pixel_counts(train_dataset)
    val_class_counts, val_total_pixels = compute_class_pixel_counts(val_dataset)

    # Print class ratios
    print("Class Ratios Between Training and Validation Datasets:")
    for idx, cls in enumerate(class_indices):
        train_count = train_class_counts.get(cls, 0)
        val_count = val_class_counts.get(cls, 0)

        # Avoid division by zero
        if val_count == 0:
            ratio = "Undefined (division by zero)"
        else:
            ratio = train_count / val_count

        class_name = f"Class {cls}"  # You can modify this to use actual class names
        print(f"{class_name} (Index {cls}):")
        print(f"  Training Pixels: {train_count}")
        print(f"  Validation Pixels: {val_count}")
        print(f"  Ratio (Train/Validation): {ratio}\n")


def compute_per_class_metrics(model: torch.nn.Module,
                              train_dataset: Dataset,
                              val_dataset: Dataset,
                              class_colors: dict,
                              device: torch.device,
                              output_path: str) -> dict:
    """
    Compute per-class Dice Loss and IoU metrics for a model on both training and validation datasets.

    :param model: model
    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param class_colors: class colors dictionary
    :param device: device
    :param output_path: path to save the metrics

    :return: Dictionary containing per-class Dice Loss and IoU metrics for training and validation datasets
    """
    cls_indices = list(i for i in range(len(class_colors)))
    def compute_metrics_for_dataset(dataset: Dataset, dataset_name: str) -> tuple[dict, dict]:
        nonlocal cls_indices
        per_class_dice_loss = {cls_idx: [] for cls_idx in cls_indices}
        per_class_iou = {cls_idx: [] for cls_idx in cls_indices}

        # Define loss function (assuming you have MultiClassDiceLoss defined)
        dice_loss_fn = MultiClassDiceLoss(mode='multiclass', from_logits=True, reduction='none')

        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset_name} dataset"):
                image, mask, *_ = dataset[idx][:2]  # Get image and mask
                image = image.unsqueeze(0).to(device)  # Add batch dimension
                mask = mask.to(device)

                # Forward pass
                output = model(image)  # Shape: (1, C, H, W)

                # Convert mask to class index mask
                if not mask.ndim == 2:
                    if mask.shape[0] != 3:
                        raise RuntimeError(f"Mask should be a 2D tensor of size (H, W) or a 3D tensor of size (C, H, W), got {mask.shape}")
                    mask = rgb_to_mask(mask, class_colors).to(device)  # Shape: (H, W)

                # Get unique classes in the ground truth mask
                unique_classes = torch.unique(mask)

                # Create one-hot encoded masks
                num_classes = len(class_colors)
                mask_one_hot = F.one_hot(mask.long(), num_classes=num_classes).permute(2, 0, 1)  # Shape: (C, H, W)
                mask_one_hot = mask_one_hot.unsqueeze(0)  # Shape: (1, C, H, W)

                # Compute Dice Loss per class
                dice_loss = dice_loss_fn(output, mask_one_hot)  # Shape: (C,)

                # Compute IoU per class
                preds = torch.argmax(output, dim=1)  # Shape: (1, H, W)
                preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)  # Shape: (1, C, H, W)

                intersection = (preds_one_hot & mask_one_hot).sum(dim=(2, 3)).squeeze(0).float()  # Shape: (C,)
                union = (preds_one_hot | mask_one_hot).sum(dim=(2, 3)).squeeze(0).float()  # Shape: (C,)
                iou = intersection / (union + 1e-7)  # Avoid division by zero

                # For each class in the ground truth, append the metrics
                for cls_idx in unique_classes:
                    cls_idx = int(cls_idx.item())
                    per_class_dice_loss[cls_idx].append(dice_loss[cls_idx].item())
                    per_class_iou[cls_idx].append(iou[cls_idx].item())

        # Compute average metrics per class
        avg_dice_loss = {cls_idx: (sum(losses) / len(losses)) if losses else None
                         for cls_idx, losses in per_class_dice_loss.items()}
        avg_iou = {cls_idx: (sum(ious) / len(ious)) if ious else None
                   for cls_idx, ious in per_class_iou.items()}

        return avg_dice_loss, avg_iou

    # Compute metrics for training dataset
    train_avg_dice_loss, train_avg_iou = compute_metrics_for_dataset(train_dataset, "training")

    # Compute metrics for validation dataset
    val_avg_dice_loss, val_avg_iou = compute_metrics_for_dataset(val_dataset, "validation")

    # Create a combined dictionary for both training and validation metrics
    metrics = {
        'training': {
            'dice_loss': train_avg_dice_loss,
            'iou': train_avg_iou
        },
        'validation': {
            'dice_loss': val_avg_dice_loss,
            'iou': val_avg_iou
        }
    }

    # Optionally create heatmaps
    def create_heatmaps(train_metrics, val_metrics, class_names):
        nonlocal cls_indices
        # Prepare data for heatmap
        metrics = ['Dice Loss', 'IoU']
        data = []

        for cls_idx in cls_indices:
            row = {'Class': class_names[cls_idx]}
            # Training metrics
            row['Train Dice Loss'] = train_metrics['dice_loss'].get(cls_idx, None)
            row['Train IoU'] = train_metrics['iou'].get(cls_idx, None)
            # Validation metrics
            row['Validation Dice Loss'] = val_metrics['dice_loss'].get(cls_idx, None)
            row['Validation IoU'] = val_metrics['iou'].get(cls_idx, None)
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index('Class', inplace=True)

        # Create heatmaps
        for metric in metrics:
            heatmap_data = df[[f'Train {metric}', f'Validation {metric}']]
            plt.figure(figsize=(8, len(cls_indices)))
            sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='viridis')
            plt.title(f'Per-Class {metric} for Training and Validation')
            plt.ylabel('Classes')
            plt.xlabel('Dataset')
            plt.tight_layout()
            plt.savefig(f'models/torch_model_files/{model.__class__.__name__}_{metric}_heatmap.png')
            plt.show()

    create_heatmaps(metrics['training'], metrics['validation'], cls_indices)

    save_json(output_path, metrics)

    return metrics

def compute_and_save_ssim_matrices(dataset,
                                   output_dir: str,
                                   batch_size: int = 1,
                                      grayscale: bool = False,
                                   sigma: float = None,
                                   compression_quality: float = None) -> None:
    """
    Compute and save SSIM and MS-SSIM matrices for the given dataset. Then, save the result as a .pt file.
    In the output matrix for ssim and ms_ssim, only the upper triangular part is computed (otherwise, the code would
    take too long to run). The matrices are then made symmetric by adding the transpose of the upper triangular part.

    The kernel size is determined using the empirical rule:
    - The kernel radius should span 3 times the standard deviation: kernel_radius = int(3 * sigma)
    - The kernel size is 2 * kernel_radius + 1

    :param dataset: dataset object
    :param output_dir: output directory for the .pt files
    :param batch_size: batch size
    :param sigma: standard deviation for Gaussian blur
    :param compression_quality: quality for image compression

    :raises ValueError: if dataset is not transformed using TRANSFORMER from config.py
    """
    if dataset.transform != TRANSFORMER:
        raise ValueError("Please transform your dataset using the TRANSFORMER from config.py")

    num_imgs = len(dataset)
    kernel_size = None

    if sigma:
        assert sigma > 0, "Sigma must be a positive value."
        kernel_size = 2 * int(3 * sigma) + 1  # Empirical rule
        extra_transforms = [transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)] if not grayscale \
                        else [transforms.Grayscale(num_output_channels=1), transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)]
        new_transforms = transforms.Compose(dataset.transform.transforms + extra_transforms)
        dataset.transform = new_transforms
        print(f"Kernel size used for sigma={sigma}: {kernel_size}")

    images = [image_array for image_array, *_ in dataset]
    image_paths = [os.path.basename(path) for *_, path in dataset]
    images = torch.stack(images, dim=0).to('cuda')

    indices = torch.triu_indices(num_imgs, num_imgs, offset=1, device='cuda')
    rows, cols = indices[0], indices[1]
    num_pairs = rows.size(0)
    ssim_result = torch.zeros((num_imgs, num_imgs), device='cuda')
    ms_ssim_result = torch.zeros((num_imgs, num_imgs), device='cuda')
    for batch_start in tqdm(range(0, num_pairs, batch_size), desc=f'Computing SSIM/MS-SSIM:'):
        batch_end = min(batch_start + batch_size, num_pairs)
        batch_rows = rows[batch_start:batch_end]
        batch_cols = cols[batch_start:batch_end]
        img_rows = images[batch_rows]
        img_cols = images[batch_cols]
        ssim_score = ssim(img_rows, img_cols, data_range=1.0, reduction='none')
        ms_ssim_score = ms_ssim(img_rows, img_cols, data_range=1.0, reduction='none')

        ssim_result[batch_rows, batch_cols] = ssim_score
        ms_ssim_result[batch_rows, batch_cols] = ms_ssim_score

    # Make the matrices symmetric
    ssim_result = ssim_result + ssim_result.T + torch.eye(num_imgs, device='cuda')
    ms_ssim_result = ms_ssim_result + ms_ssim_result.T + torch.eye(num_imgs, device='cuda')
    save_to_hdf5(f'{output_dir}/ssim_sigma{sigma}.h5',
                 {'ssim': ssim_result.cpu().numpy(),
                            'ms_ssim': ms_ssim_result.cpu().numpy(),
                            'image_paths': image_paths})
    print("Saved SSIM and MS-SSIM matrices using: \n"
          f"sigma={sigma}, kernel_size={kernel_size}, compression_quality={compression_quality}")


def compute_and_save_ssim_matrices_train_val(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    grayscale: bool = False,
    sigma: float = None,
    batch_size: int = 1
) -> None:
    """
    Compute SSIM and MS-SSIM matrices for all pairs between val_dataset and train_dataset.

    :param train_dataset Train dataset
    :param val_dataset: Validation dataset
    :param output_dir: Directory to save the results
    :param sigma: optional Gaussian blur sigma
    :param batch_size: batch size for processing
    """
    print(f"Computing ssim and ms_ssim with sigma={sigma} for all pairs (val vs train).")
    if not train_dataset.transform == val_dataset.transform:
        raise ValueError("Train and validation datasets must have the same transform.")
    kernel_size = None
    if sigma:
        assert sigma > 0, "Sigma must be a positive value."
        kernel_size = 2 * int(3 * sigma) + 1
        extra_transforms = [transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)] if not grayscale \
                        else [transforms.Grayscale(num_output_channels=1), transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)]
        new_transforms = transforms.Compose(train_dataset.transform.transforms + extra_transforms)
        train_dataset.transform = new_transforms
        val_dataset.transform = new_transforms
        print(f"Kernel size used for sigma = {sigma}: {kernel_size}")

    num_val = len(val_dataset)
    num_train = len(train_dataset)

    all_val_paths = [os.path.basename(path) for *_, path in val_dataset]
    print("All validation paths loaded.")
    all_train_paths = [os.path.basename(path) for *_, path in train_dataset]
    print("All training paths loaded.")

    ssim_arr = torch.zeros((num_val, num_train), device=DEVICE)
    ms_ssim_arr = torch.zeros((num_val, num_train), device=DEVICE)

    hdf5_path = f'{output_dir}/ssim_sigma{sigma}.h5'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for i_val in tqdm(range(num_val), desc="Computing SSIM/MS-SSIM (val vs train)"):
        val_img, *_, val_path = val_dataset[i_val]
        val_img = val_img.unsqueeze(0).to(DEVICE).clip(0, 1)  # (1, C, H, W)
        current_train_idx = 0

        for train_batch, *_, train_paths in train_loader:
            train_batch = train_batch.to(DEVICE).clip(0, 1)
            val_batch = val_img.repeat(train_batch.size(0), 1, 1, 1).to(DEVICE).clip(0, 1)

            ssim_batch = ssim(val_batch, train_batch, data_range=1.0, reduction='none')
            ms_ssim_batch = ms_ssim(val_batch, train_batch, data_range=1.0, reduction='none')
            batch_size_actual = ssim_batch.size(0)
            ssim_arr[i_val, current_train_idx:current_train_idx + batch_size_actual] = ssim_batch
            ms_ssim_arr[i_val, current_train_idx:current_train_idx + batch_size_actual] = ms_ssim_batch
            #### DEBUG####
            ssim_arr_np = ssim_arr.cpu().numpy()
            ms_ssim_arr_np = ms_ssim_arr.cpu().numpy()
            #### DEBUG####
            current_train_idx += batch_size_actual

        #### DEBUG####
        # if i_val == 10: # Break to check if hdf5 file is created properly
        #     break
        #### DEBUG####
    #### DEBUG####
    ssim_arr_np = ssim_arr.cpu().numpy()
    ms_ssim_arr_np = ms_ssim_arr.cpu().numpy()
    #### DEBUG####

    save_to_hdf5(hdf5_path, {'val_paths': all_val_paths,
                                'train_paths': all_train_paths,
                                'ssim': ssim_arr.cpu().numpy(),
                                'ms_ssim': ms_ssim_arr.cpu().numpy()})
    print(f"Saved train-val SSIM and MS-SSIM matrices at {output_dir} with sigma={sigma}, kernel_size={kernel_size}.")
    # with h5py.File(hdf5_path, 'w') as f:
    #     for i_val, (val_img, *_, val_path) in enumerate(tqdm(val_dataset, desc="Computing SSIM/MS-SSIM (val vs train)")):
    #         val_img = val_img.unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    #         for train_batch, *_, train_paths in train_loader:
    #             train_batch = train_batch.to(DEVICE).clip(0, 1)  # (batch_size, C, H, W)
    #             val_batch = val_img.repeat(train_batch.size(0), 1, 1, 1).to(DEVICE).clip(0, 1)  # (batch_size, C, H, W)
    #
    #             ssim_batch = ssim(val_batch, train_batch, data_range=1.0, reduction='none')  # (batch_size,)
    #             ms_ssim_batch = ms_ssim(val_batch, train_batch, data_range=1.0, reduction='none')  # (batch_size,)
    #
    #             for i_train, ssim_scores, ms_ssim_scores, train_path in zip(
    #                 range(len(train_batch)), ssim_batch, ms_ssim_batch, train_paths
    #             ):
    #                 key = f"{os.path.basename(val_path)}@@{os.path.basename(train_path)}"
    #                 grp = f.create_group(key)
    #                 grp.create_dataset('ssim', data=ssim_scores.cpu().numpy())
    #                 grp.create_dataset('ms_ssim', data=ms_ssim_scores.cpu().numpy())
    # print(f"Saved train-val SSIM and MS-SSIM matrices at {output_dir} with sigma={sigma}, kernel_size={kernel_size}.")


def compute_and_save_confidence_vectors(model: _BaseSegmentationModel,
                                        dataset: ExcavatorDataset,
                                        only_classes_in_gt_mask=False,
                                        ignore_background=False,
                                        output_dir: str = None) -> None:
    """
    Compute and save prediction probability vectors for the dataset.

    **Note**: Use on the train dataset only! (Corrent stand of the project: 20.12.2024)

    If `only_classes_in_gt_mask` is True, the data will be saved liek this:

    ```
    {
        'image1.images': {
            'classes': [0, 1, 2, 3, 4],
            'confidence': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'image2.images': {
            'classes': [0, 1, 2, 3, 4],
            'confidence': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        ...
    }
    ```

    If `only_classes_in_train_set` is False, the data will be saved like this:

    ```
    {
        'image1.images': [0.1, 0.2, 0.3, 0.4, 0.5],
        'image2.images': [0.1, 0.2, 0.3, 0.4, 0.5],
        ...
    }

    :param model: Segmentation model
    :param dataset: Dataset
    :param only_classes_in_gt_mask: If True, only classes in the train set will be considered
    :param ignore_background: If True, the background class will be ignored
    :param output_dir: Output directory
    """
    model_name = model.model.__class__.__name__

    class_colors = dataset.class_colors

    if only_classes_in_gt_mask:
        print("Only classes in the ground truth mask will be considered.")
        train_results = defaultdict(lambda: {'classes': [], 'confidence': []})
        for img, mask, path in tqdm(dataset, desc=f"Computing confidence vectors in Dataset for {model_name}"):
            conf_with_bg, _ = model.predict_single_image(img,
                                                         mask,
                                                         return_raw_prob_vector=True,
                                                         ignore_background=ignore_background)
            train_results[os.path.basename(path)]['classes'] = get_class_idx(mask, class_colors=class_colors)
            train_results[os.path.basename(path)]['confidence'] = conf_with_bg.cpu().numpy().tolist()
    else:
        print("All classes in the dataset will be considered.")
        train_results = {}
        for img, mask, path in tqdm(dataset, desc=f"Computing confidence vectors in Dataset for {model_name}"):
            classes_of_interest = get_class_idx(mask, class_colors=class_colors) if only_classes_in_gt_mask else []
            conf_with_bg, _ = model.predict_single_image(img,
                                                         mask,
                                                         return_raw_prob_vector=True,
                                                         cls_of_interest=classes_of_interest,
                                                         ignore_background=ignore_background)
            train_results[os.path.basename(path)] = conf_with_bg.cpu().numpy()

    save_to_hdf5(output_dir, train_results)