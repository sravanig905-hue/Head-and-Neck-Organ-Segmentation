HaN-Seg metrics fix

Replace your project files:
  backend/app.py -> this backend/app.py
  frontend/script.js -> this frontend/script.js

The backend now loads:
  hybrid_unet_transformer_multiorgan_best.pth

The /segment_nrrd response includes test_set_metrics loaded from evaluation_metrics.json.
The frontend dashboard cards use those genuine test-set values instead of current-slice metrics.

Restart backend and hard-refresh the browser after replacing the files.
