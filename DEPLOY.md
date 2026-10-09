# Deploying the API on AWS (free credits)

## Option A: EC2 + Docker (about 30 minutes)
1. AWS console → EC2 → Launch instance: Ubuntu 24.04, `t3.micro`, new key pair.
   Security group: allow SSH (22) from your IP and TCP 8000 from anywhere.
2. Connect and install Docker:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
   sudo usermod -aG docker $USER && newgrp docker
   ```
3. Clone and start:
   ```bash
   git clone https://github.com/AdityaTiwarime/churn-intelligence.git
   cd churn-intelligence
   echo "OPENAI_API_KEY=sk-..." > .env       # optional, enables LLM retention plans
   docker compose up -d --build
   ```
4. Open `http://<EC2-public-IP>:8000/docs`. That is the live link for the resume and README.

Stop the instance when you are not demoing, so credits last.

## Option B: Render (free, link never expires)
New → Web Service → connect the GitHub repo → Runtime: Docker → instance type Free → Deploy.
Render builds the `Dockerfile` and gives a permanent `https://<name>.onrender.com/docs` URL.
(Free instances sleep when idle; the first request takes ~30 seconds.)
