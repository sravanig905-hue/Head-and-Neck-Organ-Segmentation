HaN-Seg AI - FINAL DYNAMIC WEBSITE

1. Keep this folder merged into the existing HaN_Seg_Project root.
2. Required existing files:
   - model/hybrid_unet_transformer.py
   - hybrid_unet_transformer_multiorgan.pth
   - HaN-Seg/HaN-Seg/set_1/
   - multi_organ_cache/ (optional but recommended for faster ground truth lookup)
3. Start backend from the project root:
   .venv312\Scripts\Activate.ps1
   python backend\app.py
4. Open frontend with Live Server:
   http://127.0.0.1:5500/frontend/index.html

Flow:
Upload CT NRRD -> display real uploaded slice -> select slice -> Analyze ->
Hybrid U-Net + Transformer -> 30-organ prediction -> original/mask/overlay ->
OAR boundary/location/pixel area -> matching ground truth -> scan-specific metrics.

Metrics are NOT hard-coded. Dice, IoU, Precision, Recall and Accuracy are calculated
for the analyzed slice when matching HaN-Seg ground truth is available.
For a CT without ground truth, segmentation/boundary output can still be generated,
but ground-truth comparison metrics correctly show unavailable.

Accuracy target 0.9600 is displayed as a project target only; it is not claimed as
an actual measured result unless the uploaded scan calculation produces 0.9600.
