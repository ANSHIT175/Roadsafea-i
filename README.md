# RoadSafe AI

## Deployment and model loading

- The trained `best.pt` file is stored as a public GitHub Release asset, not inside the Vercel Function bundle.
- The backend downloads the checkpoint on its first inference request from:
  `https://github.com/ANSHIT175/Roadsafea-i/releases/download/v1.0.0/best.pt`
- `MODEL_URL` can be set as a Vercel environment variable to override the default URL.
- For local development, a valid `models/best.pt` is used automatically when present; otherwise the Release asset is downloaded to `/tmp`.
- The downloaded checkpoint is validated before use, and the download is written atomically to avoid using partial files.
- The backend converts the checkpoint into an ONNX model in `/tmp` and runs inference with ONNX Runtime. PyTorch and Ultralytics are not bundled into the Vercel Function.

## Inference integrity and limitations

- Every uploaded image is sent to the server-side model. The browser does not fabricate replacement detections when the API is unavailable.
- Risk is calculated from model detections, confidence, and bounding-box area with diminishing contribution from additional detections.
- NMS removes near-contained duplicate boxes while keeping boxes from different classes separate.
- A zero-detection result is not proof that the road is defect-free. It should be manually reviewed, especially for distant, low-resolution, dark, or partially obscured road surfaces.

## Detection refinement notes

- Inference uses the checkpoint's one-to-many detection head.
- Crack confidence and minimum-box gates are less restrictive; foliage/sky rejection requires most pixels in a box to match those backgrounds.
- The risk score uses a bounded per-hazard increment to reduce score saturation from multiple boxes.
- These are post-processing and inference-path refinements, not model retraining. Detection accuracy and risk thresholds still need validation against a labelled set of clean, cracked, and pothole road images.
