# Retinal Blood Vessel Segmentation with PyTorch

An educational implementation of U-Net for segmenting blood vessels in retinal fundus images from the DRIVE dataset.

## Project structure

```text
.
|-- UNET/
|   |-- data.py       # Dataset and train/validation split
|   |-- model.py      # U-Net architecture
|   |-- loss.py       # Dice loss and metrics
|   |-- utils.py      # Reproducibility and checkpoints
|   |-- train.py      # Training entry point
|   `-- test.py       # Inference and evaluation entry point
|-- data_aug.py       # Paired image/mask augmentations
|-- data/DRIVE/       # Download separately; ignored by Git
|-- checkpoints/      # Saved model weights
|-- results/          # Predicted masks
|-- requirements.txt
`-- README.md
```

## Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, use the official PyTorch installation selector first.

## Add the DRIVE dataset

Download DRIVE from its official source and arrange it as follows. Do not upload the dataset to GitHub:

```text
data/DRIVE/
|-- training/
|   |-- images/
|   `-- 1st_manual/
`-- test/
    |-- images/
    `-- 1st_manual/
```

The loader pairs sorted image and mask files, so each split must contain the same number of images and masks.

## Train and test

```bash
python -m UNET.train --data_dir data/DRIVE --epochs 50 --batch_size 2
python -m UNET.test --data_dir data/DRIVE --checkpoint checkpoints/best_model.pth
```

The best model is saved to `checkpoints/best_model.pth`; predicted masks are saved to `results/predictions/`.

## Deploy as a web app

After training, start the local deployment with:

```bash
streamlit run app.py
```

Open the URL shown in the terminal, usually `http://localhost:8501`. Upload a retinal image and the app will show the input, predicted vessel mask, and overlay.

For Streamlit Community Cloud:

1. Push this project to GitHub.
2. Push `checkpoints/best_model.pth` as well, or store it with Git LFS.
3. Go to [share.streamlit.io](https://share.streamlit.io), choose your repository, and set the main file to `app.py`.
4. Streamlit will install `requirements.txt` and launch the app.

The model checkpoint is ignored by default because trained weights can be large. If you need it in the Git repository, remove the `checkpoints/*` line from `.gitignore` before committing it.

## Upload to GitHub

Create an empty repository on GitHub, then run:

```bash
git add .
git commit -m "Build retinal vessel segmentation pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

This is an independent implementation inspired by the public reference repository. Keep the original repository link and the DRIVE dataset citation in your documentation.
