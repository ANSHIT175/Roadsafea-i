# RoadSafe AI

## Inference integrity and model files

- `models/best.pt` is included in this package and is loaded by the FastAPI backend.
- Every uploaded image and bundled sample image is sent to the server-side model.
- The browser does not generate substitute detections or confidence values when the API is unavailable. It shows an error instead.
- Risk is calculated from model detections, confidence, and bounding-box area with diminishing contribution from additional detections.
- NMS also removes near-contained duplicate boxes.

A zero-detection result means the model returned no detections for that image; it is not a guarantee that the road is defect-free. Model performance still depends on the training data and should be validated against labelled road images before operational use.
