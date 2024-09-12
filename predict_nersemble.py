from tqdm import tqdm
import argparse
from pathlib import Path
from glob import glob
import pandas as pd
import torch
import os
import numpy as np

from batch_predict import Dataset, load_model


def do_seq(smirk_encoder, seq_path, device, no_crop, image_size):
    impaths = sorted(glob(f"{seq_path}/images/*.png"))
    dataset = Dataset(impaths, no_crop, image_size)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
    )

    print(f"Predicting {len(dataset)} images")

    res = []
    with torch.no_grad():
        for cropped_image, path, bad in tqdm(dataloader):
            cropped_image = cropped_image.to(device)

            outputs = smirk_encoder.expression_encoder(cropped_image)
            outputs_pose = smirk_encoder.pose_encoder(cropped_image)
            score = (
                outputs_pose["pose_params"][:, 0]
                + outputs_pose["pose_params"][:, 1] * 2
            ).abs()

            for i, p in enumerate(path):
                timestep, cam_id = Path(p).name.split("_")
                res.append(
                    [
                        timestep,
                        cam_id,
                        outputs["expression_params"][i].cpu().numpy(),
                        outputs["eyelid_params"][i].cpu().numpy(),
                        outputs["jaw_params"][i].cpu().numpy(),
                        score[i].item() if not bad[i] else 10000.0,
                    ]
                )

    df = pd.DataFrame(
        res, columns=["timestep", "cam_id", "expression", "eyelid", "jaw", "score"]
    )

    # for each timestep, rank the scores from lowest to highest
    df["rank"] = df.groupby("timestep")["score"].rank()

    # average the expression code for the lowest 3 ranked items, for each timestep
    per_timestep = {}
    for k in ["expression", "eyelid", "jaw"]:
        per_timestep[k] = df.groupby("timestep").apply(
            lambda x: x[x["rank"] <= 3][k].mean(axis=0)
        )

    print("Saving new flame params npz's to {flame_params_dir}/_smirk/...")
    flame_params_dir = f"{seq_path}/flame_param/"
    for npz_path in tqdm(sorted(glob(f"{flame_params_dir}/*.npz"))):
        timestep_npz = Path(npz_path).name.split(".")[0]
        if os.path.exists(f"{flame_params_dir}_smirk/{timestep_npz}.npz"):
            print(f"Skipping {timestep_npz}.npz")
            continue
        print(npz_path)
        flame_param = np.load(npz_path)
        flame_param = {k: v for k, v in flame_param.items()}

        flame_param["expr"] = np.zeros((flame_param["expr"].shape[0], 102))
        flame_param["expr"][:, :50] = per_timestep["expression"][timestep_npz]
        flame_param["expr"][:, 100:102] = per_timestep["eyelid"][timestep_npz]

        flame_param["jaw_pose"] = np.zeros_like(flame_param["jaw_pose"])
        flame_param["jaw_pose"][:] = per_timestep["jaw"][timestep_npz]

        Path(f"{flame_params_dir}_smirk").mkdir(exist_ok=True, parents=True)

        np.savez(f"{flame_params_dir}_smirk/{timestep_npz}.npz", **flame_param)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=int, help="Subject to process (e.g. '306')")
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
    args = parser.parse_args()

    print(f"Processing subject {args.subject}")

    image_size = 224

    smirk_encoder = load_model(args.checkpoint, args.device)

    seq_paths = glob(
        f"/home/shared/shenhan-cvpr-data/nersemble_masked/{args.subject}/{args.subject}*Line"
    )
    for seq_path in seq_paths:
        print(f"Processing {seq_path}")
        do_seq(smirk_encoder, seq_path, args.device, args.no_crop, image_size)


if __name__ == "__main__":
    main()
