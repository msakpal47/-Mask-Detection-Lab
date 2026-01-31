# Deployment Guide for Mask Detection App

## Project Structure
The project has been restructured for easier deployment:

- **app.py**: Main FastAPI application entry point.
- **streamlit_app.py**: Streamlit version of the application.
- **config.py**: Configuration settings.
- **models/**: Contains the trained model (`mask_detector.h5`).
- **web/**: Contains the frontend interface (`index.html`).
- **scripts/**: Utility scripts for training and data processing.
- **requirements.txt**: Python dependencies.

## deploying on AWS EC2

### 1. Launch EC2 Instance
- Launch an **Ubuntu 22.04** (or similar) instance.
- Ensure **Security Group** allows inbound traffic on port `8000` (for FastAPI) or `8501` (for Streamlit), and `22` (SSH).

### 2. Connect via Putty
- Convert your `.pem` key to `.ppk` using PuTTYgen if needed.
- Open Putty, enter the Public IP of your EC2 instance.
- Under **Connection > SSH > Auth**, browse and select your `.ppk` key.
- Login as `ubuntu`.

### 3. Transfer Files via WinSCP
- Open WinSCP.
- Host name: Your EC2 Public IP.
- User name: `ubuntu`.
- Advanced > SSH > Authentication > Private key file: Select your `.ppk` key.
- Connect.
- Drag and drop the entire `Mask_detection` folder (excluding `archive` and `venv` if you want to save space) to `/home/ubuntu/`.

### 4. Install Dependencies on EC2
Run the following commands in Putty:

```bash
sudo apt update
sudo apt install python3-pip python3-venv libgl1 -y

# Navigate to the project folder
cd Mask_detection

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 5. Run the Application

#### Option A: Run FastAPI Backend
```bash
python3 app.py
```
- The app will run on `http://0.0.0.0:8000`.
- Access it via `http://<EC2-Public-IP>:8000`.

#### Option B: Run Streamlit App
```bash
streamlit run streamlit_app.py
```
- Access it via `http://<EC2-Public-IP>:8501`.

### 6. Keep App Running (Optional)
To keep the app running after closing Putty, use `nohup` or `tmux`.

```bash
nohup python3 app.py > app.log 2>&1 &
```

## Notes
- The model file is located in `models/mask_detector.h5`.
- The interface index file is in `web/index.html`.
- Unrequired files have been moved to `archive/` or `scripts/` to keep the root clean.
