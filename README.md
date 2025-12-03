<img src="https://img.icons8.com/fluency/96/health-graph.png" width="120" align="right" />

# Smart Health Guardian  
### Real-Time Intelligent Health Monitoring System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  
[![Stars](https://img.shields.io/github/stars/your-username/smart-health-guardian?style=social)](https://github.com/your-username/smart-health-guardian)

> A modern web application based on **Streamlit + MediaPipe + YOLOv8** that automatically detects risky health behaviors:  
> - Uncovered sneezes  
> - Incorrect mask wearing  
> - Social distancing violations  

Ideal for hospitals, schools, offices, train stations, airports, etc.

---

## Demo Screenshots

Here are some screenshots showcasing the main features of Smart Health Guardian:

<p float="left">
  <img src="smart-health_guardian/Capture_114129.png" width="250" />
  <img src="demo/screenshots/Capture_114245.png" width="250" />
  <img src="demo/screenshots/Capture_114354.png" width="250" />
</p>

<p float="left">
  <img src="demo/screenshots/Capture_114435.png" width="250" />
  <img src="demo/screenshots/Capture_114504.png" width="250" />
  <img src="demo/screenshots/Capture_114629.png" width="250" />
</p>

---

## Key Features

| Module                     | Feature                                                                 | Technology Used                     |
|----------------------------|-------------------------------------------------------------------------|-------------------------------------|
| Sneezes Detection          | Detects sneezes + checks if the person covers mouth/nose (hand, elbow) | MediaPipe Pose + Hands               |
| Mask Detection             | With mask / Without mask / Incorrect mask                              | Custom-trained YOLOv8                |
| Social Distancing          | Alerts in real-time when two people are too close                      | YOLOv8n + Euclidean centroid distance |
| Modern Interface           | Smooth, responsive design with dark gradient and animations           | Streamlit + Custom CSS               |
| Export Results             | Annotated video + detailed JSON report                                 | OpenCV + JSON                        |

---

## Quick Installation (Local)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-health-guardian.git
cd smart-health-guardian

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py


