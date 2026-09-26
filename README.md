# Passport Stamp & Visa Capture API

An open-source, local microservice engineered to extract arrival dates, ports of entry, visa numbers, and handwritten consular annotations from scanned or photographed passport pages[cite: 1, 2]. 

This service feeds directly into the Bureau of Immigration **Form-C** arrival report requirements for homestays and hospitality establishments (specifically Field 4: *Place of issue of visa*, Field 5: *Date of Arrival in India*, and Field 12: *Country & city you arrived to India*)[cite: 1, 2].

---

## 1. Architectural Evolution & Tech Stack History

#### The Problem with Traditional OCR
Standard passport bio-data pages adhere to strict ICAO 9303 standards with Machine Readable Zones (MRZ). However, immigration stamp pages and visa pages are unstructured and noisy:
- **Arrival / Landing Stamps:** Faint ink, skewed or upside-down orientations, overlapping stamps, and diverse fonts.
- **Visa Pages:** Mix of machine-printed text and handwritten officer endorsements, issue stamps, or consular notes.

Traditional OCR engines (such as Tesseract or mobile ML Kit) produce disjointed text tokens without structural context, failing to distinguish between an entry stamp and an exit stamp on the same page[cite: 1].

#### Why Not Cloud APIs?
While commercial multimodal vision APIs (like Google Gemini Vision) perform well, they incur recurring per-call fees and require an active internet connection. The objective was a **100% free, privacy-first, on-premise, open-source solution**.

#### Model Selection: Qwen2-VL-2B-Instruct
We selected **Qwen2-VL-2B-Instruct** running via Hugging Face Transformers:
1. **Visual Token Architecture:** It handles native aspect ratios and rotated text without fixed image squashing.
2. **Lightweight Footprint:** At 2.2 billion parameters, it requires only ~4.5 GB of RAM in 32-bit float mode, allowing deterministic vision reasoning directly on a standard CPU.
3. **Consolidated Engine:** Instead of loading multiple models or running distinct server instances for visas and stamps (which would multiply RAM requirements), a single loaded model powers both endpoints dynamically.

---

## 2. Service Specifications

#### Endpoint 1: Arrival Stamp Extraction
- **Route:** `POST /api/v1/extract/stamp`
- **Purpose:** Identifies arrival/entry stamps while ignoring departure/exit stamps, extracting the landing date and port of entry[cite: 1, 2].
- **Input:** Multipart form upload (`file`: image/jpeg, image/png).
- **Output Schema:**
```json
{
  "is_stamp_detected": true,
  "arrival_date": "2024-10-14",
  "port_of_entry": "COCHIN AIRPORT",
  "stamp_type": "ENTRY",
  "stay_permitted_until": null,
  "confidence_note": null
}
```

#### Endpoint 2: Visa Details Extraction
- **Route:** POST /api/v1/extract/visa
- **Purpose:** Detects sticker visas, e-Visa printouts, or consular rubber stamps to extract the visa number, place of issue, dates, and manual handwritten endorsements[cite: 1, 2].
- **Input:** Multipart form upload (file: image/jpeg, image/png).
- **Output Schema:**

```json
{
  "is_visa_detected": true,
  "visa_number": "V1234567",
  "place_of_issue": "LONDON",
  "date_of_issue": "2024-01-10",
  "date_of_expiry": "2025-01-09",
  "visa_type": "Tourist",
  "entries_allowed": "Multiple",
  "handwritten_notes": "Permitted stay 90 days per visit",
  "confidence_note": null
}
```

## 3. Installation Guide: Local Development (VS Code)
#### Prerequisites
- Python 3.10 or 3.11 installed.
- Minimum 8 GB RAM and 10 GB free disk space.

#### Steps
- Clone the repository:
```Bash
git clone [https://github.com/ajmanohar/PassportStampCaptureAPI.git](https://github.com/ajmanohar/PassportStampCaptureAPI.git)
cd PassportStampCaptureAPI
```
- Create and activate a virtual environment:
```Bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

- Install dependencies:
```Bash
pip install --upgrade pip
pip install -r requirements.txt
```
- Initialize environment configuration:
```Bash
cp .env.example .env
```
- Start the local service:
```Bash
python app.py
```

_Note: On first startup, Hugging Face will download the Qwen2-VL-2B model weights (~4.5 GB) into ~/.cache/huggingface/. Subsequent startups load directly from disk in seconds._

- Access Interactive Swagger Documentation:
    - Open http://127.0.0.1:8000/docs in your browser to test both endpoints.


## 4. Deployment Guide: Standalone Ubuntu Server
Follow these steps to deploy this service as a managed system daemon on an Ubuntu 22.04 / 24.04 LTS instance.

- **Step 1:** System Packages & User Setup
```Bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git
```
- **Step 2:** Clone & Prepare Application Directory
```Bash
cd /opt
sudo git clone [https://github.com/ajmanohar/PassportStampCaptureAPI.git](https://github.com/ajmanohar/PassportStampCaptureAPI.git)
sudo chown -R $USER:$USER /opt/PassportStampCaptureAPI
cd /opt/PassportStampCaptureAPI
```
- **Step 3:** Setup Virtual Environment & Install Dependencies
```Bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
- **Step 4:** Configure Systemd Service
Create a systemd unit file to ensure the service boots on system startup and restarts automatically on failures:
```Bash
sudo nano /etc/systemd/system/passport-stamp-api.service
```
Paste the following configuration:

```Ini, TOML
[Unit]
Description=Passport Stamp and Visa Capture API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/PassportStampCaptureAPI
Environment="PATH=/opt/PassportStampCaptureAPI/.venv/bin"
ExecStart=/opt/PassportStampCaptureAPI/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
_(Replace User=ubuntu with your target server username if different)._


- **Step 5:** Start & Enable the Daemon
```Bash
sudo systemctl daemon-reload
sudo systemctl enable passport-stamp-api
sudo systemctl start passport-stamp-api
```
- **Step 6:** Verify Service Status
```Bash
sudo systemctl status passport-stamp-api
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
```