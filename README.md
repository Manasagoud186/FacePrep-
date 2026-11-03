
# FacePrep

**Face-Verified Mock Interview and Communication Analysis System**

Practice. Perform. Improve.

FacePrep provides face-verified mock interviews with measurable communication analysis and machine-learning-based performance insights.
FacePrep: ML-Based Face-Verified Mock Interview and Communication Analysis System is a machine-learning-based platform designed to provide students with a realistic and structured environment for interview preparation

## Project Structure

* `frontend/`: React + Vite + TypeScript + Tailwind CSS application.
* `backend/`: FastAPI Python backend server.
* `ml/`: Machine Learning models and analysis scripts.
* `data/`: Datasets for training and testing.
* `models/`: Serialized ML models.
* `docs/`: Documentation.
* `tests/`: Automated tests.

## Prerequisites

* Node.js (v18+)
* Python 3.10+

## Getting Started

### 1. Start the Backend (FastAPI)

```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The backend listens on all network interfaces at port `8000`.

### 2. Start the Frontend (React + Vite)

Open a new terminal window:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
The frontend listens on all network interfaces at port `5173`; API requests are proxied to the backend.

## Use FacePrep from other laptops on the same Wi-Fi

1. Start both servers on the laptop that will host FacePrep, following the commands above. Keep both terminals running.
2. On the host laptop, run `ipconfig` and note its IPv4 address (for example, `192.168.1.25`).
3. Allow Python and Node.js through the host laptop's firewall on **private networks**.
4. On each other laptop connected to the same Wi-Fi, open `http://<host-ip>:5173` (for example, `http://192.168.1.25:5173`).

All laptops use the host laptop's backend and SQLite database, so keep the host online. Do not expose these development servers directly to the public internet.

### Camera and microphone access on other laptops

Browsers only allow webcam and microphone access from a secure context (HTTPS or `localhost`). The HTTP LAN address above lets other laptops view the site, but registration, face verification, and interview recording require HTTPS.

To enable HTTPS for a LAN demo, create a certificate trusted by every client laptop and containing the host's LAN IP address. In the frontend terminal, with `mkcert` installed and the host address `192.168.1.25`, create a certificate and set its paths before starting Vite:

```powershell
mkcert -cert-file faceprep-cert.pem -key-file faceprep-key.pem 192.168.1.25
$env:FACEPREP_TLS_CERT = (Resolve-Path .\faceprep-cert.pem).Path
$env:FACEPREP_TLS_KEY = (Resolve-Path .\faceprep-key.pem).Path
npm run dev -- --host 0.0.0.0 --port 5173
```

Install/trust the issuing CA certificate on every client laptop using your organization's normal certificate process, then open `https://192.168.1.25:5173`. Only distribute the public CA certificate when doing this; never share `rootCA-key.pem` or `faceprep-key.pem`. Certificate trust is required for browser camera/microphone permissions.
