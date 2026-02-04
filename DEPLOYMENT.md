# Dhivehi Transliteration - Deployment Guide

## Prerequisites on Linux Server
- Python 3.9+
- pip
- systemd (standard on most Linux distributions)

## Step 1: Setup on Linux Server

### 1.1 Copy files to server
```bash
# Copy project to server
scp -r div-transliteration/ user@server:/home/user/
```

### 1.2 Install dependencies
```bash
# SSH into server
ssh user@server

# Navigate to project
cd /home/user/div-transliteration

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Step 2: Configure systemd Service

### 2.1 Edit the service file
Update `dhivehi-transliteration.service` with server paths:
- User: Linux username
- WorkingDirectory: Full path to project
- Environment PATH: Path to Python venv/bin
- ExecStart: Full path to gunicorn in venv

### 2.2 Create log directory
```bash
sudo mkdir -p /var/log/dhivehi-transliteration
sudo chown USERNAME:USERNAME /var/log/dhivehi-transliteration
```

### 2.3 Install the service
```bash
sudo cp dhivehi-transliteration.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dhivehi-transliteration
sudo systemctl start dhivehi-transliteration
```

## Step 3: Manage the Service

```bash
# Check status
sudo systemctl status dhivehi-transliteration

# Start/Stop/Restart
sudo systemctl start dhivehi-transliteration
sudo systemctl stop dhivehi-transliteration
sudo systemctl restart dhivehi-transliteration

# View logs
sudo journalctl -u dhivehi-transliteration -f
```

## Step 4: Access
- Local: http://localhost:5001
- Network: http://SERVER_IP:5001

## Firewall (if needed)
```bash
sudo ufw allow 5001/tcp
```
