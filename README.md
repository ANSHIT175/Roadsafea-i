# RoadSafe AI

## Inference integrity and model files

- `models/best.pt` is included in this package and is loaded by the FastAPI backend.
- Every uploaded image and bundled sample image is sent to the server-side model.
- The browser does not generate substitute detections or confidence values when the API is unavailable. It shows an error instead.
- Risk is calculated from model detections, confidence, and bounding-box area with diminishing contribution from additional detections.
- NMS also removes near-contained duplicate boxes.

A zero-detection result means the model returned no detections for that image; it is not a guarantee that the road is defect-free. Model performance still depends on the training data and should be validated against labelled road images before operational use.


## Detection refinement notes

- Inference uses the checkpoint's one-to-many detection head (the accuracy-oriented default) rather than the optional one-to-one fast head.
- NMS is class-aware so overlapping boxes from different damage classes are not discarded as duplicates.
- Crack confidence and minimum-box gates are less restrictive; foliage/sky rejection now requires most pixels in a box to match those backgrounds.
- The risk score uses a bounded per-hazard increment to reduce score saturation from multiple boxes.
- A zero-detection response is explicitly marked for manual review; it must not be interpreted as proof that a road is damage-free.

These are post-processing and inference-path refinements, not model retraining. Detection accuracy and risk thresholds still need validation against a labelled set of clean, cracked, and pothole road images.
