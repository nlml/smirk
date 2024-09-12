from tqdm import tqdm
from glob import glob
from pathlib import Path
import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
import argparse
from utils.mediapipe_utils import run_mediapipe


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, DST_PTS)

    return tform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, no_crop, image_size):
        self.image_paths = image_paths
        self.no_crop = no_crop
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        kpt_mediapipe = run_mediapipe(image)
        bad = False
        if not self.no_crop:
            if kpt_mediapipe is None:
                bad = True
                cropped_image = image
                assert cropped_image.dtype == np.uint8
            else:
                kpt_mediapipe = kpt_mediapipe[..., :2]

                tform = crop_face(
                    image, kpt_mediapipe, scale=1.4, image_size=self.image_size
                )

                cropped_image = warp(
                    image, tform.inverse, output_shape=(224, 224), preserve_range=True
                ).astype(np.uint8)

                cropped_kpt_mediapipe = np.dot(
                    tform.params,
                    np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T,
                ).T
                cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).float() / 255.0
        return cropped_image, path, bad


def load_model(checkpoint, device):
    smirk_encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(checkpoint)
    checkpoint_encoder = {
        k.replace("smirk_encoder.", ""): v
        for k, v in checkpoint.items()
        if "smirk_encoder" in k
    }  # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    return smirk_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_models/SMIRK_em1.pt",
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--no-crop", action="store_true", help="Don't crop the face using mediapipe"
    )
    parser.add_argument(
        "--use_smirk_generator",
        action="store_true",
        help="Use SMIRK neural image to image translator to reconstruct the image",
    )
    parser.add_argument(
        "--render_orig",
        action="store_true",
        help="Present the result w.r.t. the original image/video size",
    )
    parser.add_argument(
        "--ims_dir",
        type=str,
        default="samples",
        help="Directory containing images to predict",
    )
    args = parser.parse_args()

    image_size = 224
    device = torch.device(args.device)

    impaths = sorted(glob(f"{args.ims_dir}/*.png"))
    impaths = [str(Path(p).resolve()) for p in impaths]
    dataset = Dataset(impaths, args.no_crop, image_size)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
    )

    smirk_encoder = load_model(args.checkpoint, args.device)

    print(f"Predicting {len(dataset)} images")

    res = {"bad": [], "path": []}
    with torch.no_grad():
        for cropped_image, path, bad in tqdm(dataloader):
            cropped_image = cropped_image.to(device)
            outputs = smirk_encoder(cropped_image)
            for k, v in outputs.items():
                if k not in res:
                    res[k] = []
                res[k].append(v.cpu().numpy())
            res["bad"].append(bad.cpu().numpy())
            res["path"] += list(path)

    for k, v in res.items():
        if k == "path":
            res[k] = np.array(v)
        else:
            res[k] = np.concatenate(v, axis=0)

    # do something with res
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
