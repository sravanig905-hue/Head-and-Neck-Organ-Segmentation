import os
import sys
import uuid
import time
import base64
import tempfile
import urllib.request
import numpy as np
import SimpleITK as sitk
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "hybrid_unet_transformer_multiorgan.pth")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "HaN-Seg", "HaN-Seg", "set_1")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "multi_organ_cache")

# ---------------------------------------------------------
# AUTO-DOWNLOAD MODEL WEIGHTS (for cloud deployment)
# ---------------------------------------------------------
MODEL_URL = os.environ.get("MODEL_URL", "")

if not os.path.exists(MODEL_PATH) and MODEL_URL:
    print(f"Model weights not found locally. Downloading from MODEL_URL...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as download_err:
        print(f"ERROR downloading model: {download_err}")

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# ---------------------------------------------------------
# CLASSES
# ---------------------------------------------------------
CLASS_NAMES = [
    "Background",
    "A_Carotid_L",
    "A_Carotid_R",
    "Arytenoid",
    "Bone_Mandible",
    "Brainstem",
    "BuccalMucosa",
    "Cavity_Oral",
    "Cochlea_L",
    "Cochlea_R",
    "Cricopharyngeus",
    "Esophagus_S",
    "Eye_AL",
    "Eye_AR",
    "Eye_PL",
    "Eye_PR",
    "Glnd_Lacrimal_L",
    "Glnd_Lacrimal_R",
    "Glnd_Submand_L",
    "Glnd_Submand_R",
    "Glnd_Thyroid",
    "Glottis",
    "Larynx_SG",
    "Lips",
    "OpticChiasm",
    "OpticNrv_L",
    "OpticNrv_R",
    "Parotid_L",
    "Parotid_R",
    "Pituitary",
    "SpinalCord",
]
NUM_CLASSES = 31

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from model.hybrid_unet_transformer import HybridUNetTransformer

    model = HybridUNetTransformer(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove possible DataParallel prefix.
    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as e:
    model = None
    MODEL_READY = False
    MODEL_ERROR = str(e)

# ---------------------------------------------------------
# IN-MEMORY VOLUME STORE
#
# IMPORTANT:
# Upload endpoint does NOT run model inference.
# Upload endpoint only reads the NRRD and returns preview.
# ---------------------------------------------------------
volumes = {}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def png_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_display(arr):
    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    # Percentile window for a useful CT preview.
    lo = np.percentile(finite, 1)
    hi = np.percentile(finite, 99)

    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1) * 255
    return out.astype(np.uint8)


def get_z_count(image):
    size = image.GetSize()  # x, y, z
    return int(size[2])


def get_xy_size(image):
    size = image.GetSize()
    return int(size[0]), int(size[1])


def get_slice_array(image, z):
    z = int(z)
    depth = get_z_count(image)

    if z < 0 or z >= depth:
        raise ValueError(f"slice_index must be between 0 and {depth - 1}")

    # Extract only ONE 2D slice. Do not convert the entire 3D volume.
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([image.GetSize()[0], image.GetSize()[1], 0])
    extractor.SetIndex([0, 0, z])

    slice_image = extractor.Execute(image)
    return sitk.GetArrayFromImage(slice_image).astype(np.float32)


def slice_to_preview(arr):
    return png_base64(Image.fromarray(normalize_display(arr), mode="L"))


def resize_for_model(arr):
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)

    # Per-slice standardization for fast inference.
    mean = tensor.mean()
    std = tensor.std()

    if float(std) > 1e-6:
        tensor = (tensor - mean) / std
    else:
        tensor = tensor - mean

    tensor = F.interpolate(
        tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    return tensor.to(DEVICE)


def prediction_to_original_size(prediction, height, width):
    prediction = prediction.float().unsqueeze(1)

    prediction = F.interpolate(
        prediction,
        size=(height, width),
        mode="nearest"
    )

    return prediction[:, 0].long()


def make_mask_image(mask):
    mask = np.asarray(mask, dtype=np.uint8)

    # Simple visible indexed mask.
    # Background black; organ IDs mapped to grayscale.
    values = (mask.astype(np.uint16) * 8) % 256
    values[mask == 0] = 0

    return png_base64(Image.fromarray(values.astype(np.uint8), mode="L"))


def make_overlay(ct_arr, mask):
    ct = normalize_display(ct_arr).astype(np.float32)
    rgb = np.stack([ct, ct, ct], axis=-1)

    # Fixed deterministic colors for visible OAR overlays.
    palette = [
        (255, 70, 70),
        (70, 180, 255),
        (80, 230, 150),
        (255, 190, 60),
        (190, 100, 255),
        (255, 100, 190),
        (90, 220, 220),
        (230, 230, 80),
    ]

    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 0:
            continue

        color = np.array(palette[(class_id - 1) % len(palette)], dtype=np.float32)
        region = mask == class_id

        # Fill lightly.
        rgb[region] = rgb[region] * 0.45 + color * 0.55

        # Boundary.
        ys, xs = np.where(region)
        if len(xs):
            boundary = np.zeros(region.shape, dtype=bool)
            boundary[1:, :] |= region[1:, :] != region[:-1, :]
            boundary[:-1, :] |= region[:-1, :] != region[1:, :]
            boundary[:, 1:] |= region[:, 1:] != region[:, :-1]
            boundary[:, :-1] |= region[:, :-1] != region[:, 1:]
            rgb[boundary] = color

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return png_base64(Image.fromarray(rgb, mode="RGB"))


def find_case_folder(filename):
    name = os.path.basename(filename)
    lower = name.lower()

    if "_img_ct.nrrd" not in lower:
        return None

    case_id = name[:lower.index("_img_ct.nrrd")]

    folder = os.path.join(DATASET_ROOT, case_id)
    if os.path.isdir(folder):
        return folder

    return None


def organ_details(mask):
    """
    Return organ boundary, center, image region, image side, and pixel area.

    Region:
      top third    -> Upper
      middle third -> Middle
      bottom third -> Lower

    Side:
      left third   -> Left
      middle third -> Center
      right third  -> Right

    These Left/Right labels describe the displayed IMAGE side.
    They should not be interpreted as guaranteed patient anatomical
    left/right unless the NRRD orientation is explicitly handled.
    """
    details = []

    height, width = mask.shape[:2]

    for class_id in np.unique(mask):
        class_id = int(class_id)

        if class_id <= 0 or class_id >= len(CLASS_NAMES):
            continue

        ys, xs = np.where(mask == class_id)

        if len(xs) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        cx = float(xs.mean())
        cy = float(ys.mean())

        if cy < height / 3:
            region = "Upper"
        elif cy < (2 * height) / 3:
            region = "Middle"
        else:
            region = "Lower"

        if cx < width / 3:
            side = "Left"
        elif cx < (2 * width) / 3:
            side = "Center"
        else:
            side = "Right"

        details.append({
            "class_id": class_id,
            "organ": CLASS_NAMES[class_id],
            "boundary": f"({x1},{y1}) - ({x2},{y2})",
            "location": f"center=({cx:.0f},{cy:.0f})",
            "center_x": round(cx, 1),
            "center_y": round(cy, 1),
            "region": region,
            "side": side,
            "pixel_area": int(len(xs))
        })

    return details


def locate_gt_mask(case_folder, slice_index, height, width):
    if not case_folder:
        return None

    gt = np.zeros((height, width), dtype=np.uint8)
    found = False

    for class_id in range(1, NUM_CLASSES):
        organ = CLASS_NAMES[class_id]
        path = os.path.join(
            case_folder,
            f"{os.path.basename(case_folder)}_OAR_{organ}.seg.nrrd"
        )

        if not os.path.exists(path):
            continue

        try:
            mask_image = sitk.ReadImage(path)

            if slice_index >= mask_image.GetSize()[2]:
                continue

            extractor = sitk.ExtractImageFilter()
            extractor.SetSize([mask_image.GetSize()[0], mask_image.GetSize()[1], 0])
            extractor.SetIndex([0, 0, slice_index])
            mask_slice = extractor.Execute(mask_image)

            arr = sitk.GetArrayFromImage(mask_slice)

            if arr.shape != (height, width):
                arr_img = Image.fromarray((arr > 0).astype(np.uint8) * 255)
                arr_img = arr_img.resize((width, height), Image.Resampling.NEAREST)
                arr = np.array(arr_img) > 0

            gt[arr > 0] = class_id
            found = True

        except Exception:
            continue

    return gt if found else None


def calculate_metrics(pred, gt):
    dice_values = []
    iou_values = []
    precision_values = []
    recall_values = []

    for class_id in range(1, NUM_CLASSES):
        p = pred == class_id
        g = gt == class_id

        p_count = int(p.sum())
        g_count = int(g.sum())

        # Macro metric over organs that actually occur in GT or prediction.
        if p_count == 0 and g_count == 0:
            continue

        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        dice_values.append(dice)
        iou_values.append(iou)
        precision_values.append(precision)
        recall_values.append(recall)

    total = pred.size
    accuracy = float((pred == gt).sum() / max(total, 1))

    return {
        "available": True,
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "accuracy": accuracy
    }


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/")
def home():
    return jsonify({
        "success": True,
        "project": "Head & Neck Organ Segmentation",
        "model": "Hybrid U-Net + Transformer",
        "classes": NUM_CLASSES,
        "organs": 30,
        "device": str(DEVICE),
        "model_ready": MODEL_READY,
        "status": "Backend running successfully"
    })


@app.get("/health")
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_ready": MODEL_READY,
        "device": str(DEVICE),
        "model_error": MODEL_ERROR if not MODEL_READY else ""
    })


@app.get("/organs")
def organs():
    return jsonify({
        "success": True,
        "organs": [
            {"class_id": i, "organ": CLASS_NAMES[i]}
            for i in range(1, NUM_CLASSES)
        ]
    })


@app.post("/upload_nrrd")
def upload_nrrd():
    """
    FAST UPLOAD ENDPOINT.

    This route intentionally does NOT:
      - run the neural network
      - load 30 ground-truth masks
      - calculate metrics
      - scan the complete dataset

    It only:
      1. validates the filename
      2. saves the uploaded file temporarily
      3. reads the NRRD header/volume
      4. extracts ONE middle slice
      5. returns the preview and volume ID
    """
    started = time.time()

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file field received."
        }), 400

    uploaded = request.files["file"]

    if not uploaded or not uploaded.filename:
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    filename = os.path.basename(uploaded.filename)

    if not filename.lower().endswith(".nrrd"):
        return jsonify({
            "success": False,
            "error": "Only .nrrd files are supported."
        }), 400

    if not filename.lower().endswith("_img_ct.nrrd"):
        return jsonify({
            "success": False,
            "error": "Please select the CT file ending with _IMG_CT.nrrd."
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".nrrd",
            delete=False
        ) as tmp:
            temp_path = tmp.name
            uploaded.save(temp_path)

        # Read the NRRD only once.
        image = sitk.ReadImage(temp_path)

        if image.GetDimension() != 3:
            raise ValueError(
                f"Expected a 3D CT NRRD, got dimension {image.GetDimension()}."
            )

        size = image.GetSize()
        width, height, depth = int(size[0]), int(size[1]), int(size[2])

        if depth <= 0:
            raise ValueError("The CT volume has no slices.")

        middle = depth // 2

        # Only extract the middle slice for immediate browser display.
        middle_arr = get_slice_array(image, middle)
        preview = slice_to_preview(middle_arr)

        volume_id = str(uuid.uuid4())

        case_folder = find_case_folder(filename)

        volumes[volume_id] = {
            "image": image,
            "filename": filename,
            "case_folder": case_folder,
            "width": width,
            "height": height,
            "depth": depth,
            "created": time.time(),
        }

        elapsed = time.time() - started

        return jsonify({
            "success": True,
            "volume_id": volume_id,
            "filename": filename,
            "width": width,
            "height": height,
            "total_slices": depth,
            "current_slice": middle,
            "image_base64": preview,
            "upload_processing_seconds": round(elapsed, 3)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"NRRD upload/read failed: {str(e)}"
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/get_slice")
def get_slice():
    data = request.get_json(silent=True) or {}

    volume_id = data.get("volume_id")
    slice_index = data.get("slice_index")

    if volume_id not in volumes:
        return jsonify({
            "success": False,
            "error": "Volume not found. Upload the CT again."
        }), 404

    try:
        slice_index = int(slice_index)
    except Exception:
        return jsonify({
            "success": False,
            "error": "slice_index must be an integer."
        }), 400

    try:
        volume = volumes[volume_id]
        arr = get_slice_array(volume["image"], slice_index)

        return jsonify({
            "success": True,
            "slice_index": slice_index,
            "total_slices": volume["depth"],
            "image_base64": slice_to_preview(arr)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Slice extraction failed: {str(e)}"
        }), 500


@app.post("/segment_nrrd")
def segment_nrrd():
    if not MODEL_READY:
        return jsonify({
            "success": False,
            "error": "Model is not ready: " + MODEL_ERROR
        }), 500

    data = request.get_json(silent=True) or {}

    volume_id = data.get("volume_id")
    slice_index = data.get("slice_index")

    if volume_id not in volumes:
        return jsonify({
            "success": False,
            "error": "Volume not found. Upload the CT again."
        }), 404

    try:
        slice_index = int(slice_index)
    except Exception:
        return jsonify({
            "success": False,
            "error": "slice_index must be an integer."
        }), 400

    try:
        volume = volumes[volume_id]
        image = volume["image"]

        ct_arr = get_slice_array(image, slice_index)
        height, width = ct_arr.shape

        x = resize_for_model(ct_arr)

        started = time.time()

        with torch.no_grad():
            output = model(x)

            # Some model versions can return a tuple/list.
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Expected: [B, 31, H, W]
            if output.ndim != 4:
                raise RuntimeError(
                    f"Unexpected model output shape: {tuple(output.shape)}"
                )

            prediction = torch.argmax(output, dim=1)

        elapsed = time.time() - started

        prediction = prediction_to_original_size(
            prediction,
            height,
            width
        )

        pred_mask = prediction[0].cpu().numpy().astype(np.uint8)

        details = organ_details(pred_mask)

        gt = None
        metrics = {
            "available": False,
            "dice": None,
            "iou": None,
            "precision": None,
            "recall": None,
            "accuracy": None
        }

        # Ground truth is loaded ONLY when analysis is requested.
        if volume.get("case_folder"):
            gt = locate_gt_mask(
                volume["case_folder"],
                slice_index,
                height,
                width
            )

        if gt is not None:
            metrics = calculate_metrics(pred_mask, gt)

        return jsonify({
            "success": True,
            "slice_index": slice_index,
            "total_slices": volume["depth"],
            "width": width,
            "height": height,
            "num_predicted_organs": len(details),
            "predicted_organs": details,
            "ground_truth_available": gt is not None,
            "metrics": metrics,
            "ct_base64": slice_to_preview(ct_arr),
            "mask_base64": make_mask_image(pred_mask),
            "overlay_base64": make_overlay(ct_arr, pred_mask),
            "inference_seconds": round(elapsed, 3)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Segmentation failed: {str(e)}"
        }), 500


@app.post("/clear_volume")
def clear_volume():
    data = request.get_json(silent=True) or {}
    volume_id = data.get("volume_id")

    if volume_id:
        volumes.pop(volume_id, None)

    return jsonify({
        "success": True,
        "message": "Volume cleared."
    })


@app.get("/metrics")
def metrics():
    # Do not invent model metrics.
    path = os.path.join(PROJECT_ROOT, "evaluation_metrics.json")

    if not os.path.exists(path):
        return jsonify({
            "success": True,
            "available": False,
            "message": "No evaluation_metrics.json found."
        })

    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify({
            "success": True,
            "available": True,
            "metrics": data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("HEAD & NECK ORGAN SEGMENTATION")
    print("Hybrid U-Net + Transformer")
    print("=" * 60)
    print("Project root :", PROJECT_ROOT)
    print("Model path   :", MODEL_PATH)
    print("Model ready  :", MODEL_READY)
    print("Device       :", DEVICE)

    if not MODEL_READY:
        print("MODEL ERROR  :", MODEL_ERROR)

    print("Server       : http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        threaded=True
    )
