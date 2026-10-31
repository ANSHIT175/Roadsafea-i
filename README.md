# RoadSafe AI

RoadSafe AI is a web application for detecting road damage from images. It uses a YOLO-based model to identify damage such as potholes and cracks, then displays the detection results along with a risk score.

The idea is to make basic road inspection easier and help identify road sections that may need attention.

**Live Website:** https://roadsafea-i.vercel.app

## Features

- Upload road images for analysis.
- Detect potholes and cracks using a YOLO model.
- View detected damage with annotated images.
- Calculate a risk score based on the analysis.
- View the severity of detected road damage.
- Check previous inspection results.
- View road segments on the Risk Map.
- Export inspection reports in CSV/TXT format.

## How it works

1. Upload a road image.
2. The model analyses the image.
3. The detected road damage is displayed.
4. The application calculates a risk score.
5. The result is added to the inspection records.

## Tech Stack

| Part | Technology |
|---|---|
| Backend | Python, FastAPI |
| Object Detection | YOLO |
| Model Dataset | RDD2022 |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel |

## Project Structure

```text
Roadsafea-i/
├── api/
├── backend/
├── frontend/
├── app.py
├── index.html
├── requirements.txt
├── vercel.json
└── README.md
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/ANSHIT175/Roadsafea-i.git
cd Roadsafea-i
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download `best.pt` from the [GitHub Releases](https://github.com/ANSHIT175/Roadsafea-i/releases/tag/v1.0.0) page and place it in the `models` folder, as expected by the application.

Start the application:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API status |
| POST | `/analyze` | Analyses an uploaded road image |

## Note

This project is currently a prototype. Detection results and risk scores depend on the model's performance and should be checked against actual road conditions before being used for maintenance decisions.

## Project

Developed for **Tech Eximius 2.0**.

**Repository:** https://github.com/ANSHIT175/Roadsafea-i
