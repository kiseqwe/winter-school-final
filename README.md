# Winter School 2026: Mobility AI Project 🚖🤖

This repository contains the practical assignments for the Winter School 2026 workshop. The project focuses on **Privacy-Aware AI** in the domain of Urban Mobility (Ride-Hailing services).

## 📌 Project Modules

### 1. Privacy & Compliance (DPIA)
Conducted a **Data Protection Impact Assessment** (DPIA) according to GDPR standards.
- Identified risks: Re-identification via GPS tracks, PII leakage by AI.
- Implemented controls: Geo-truncation, RLHF Alignment.
- 📄 Report: `reports/DPIA_Report.docx`

### 2. Local LLM Fine-Tuning (DPO)
Fine-tuned a **TinyLlama-1.1B** model locally (CPU-only) to act as a privacy-conscious taxi support assistant.
- **Method:** Direct Preference Optimization (DPO) + QLoRA.
- **Goal:** Prevent PII leakage (phone numbers) and maintain a professional tone.
- **Code:** `local_models/train_dpo.py`

### 3. Reinforcement Learning (RL)
"Vibe-coded" a Taxi Grid environment using LLM prompting and trained a Q-Learning agent.
- **Environment:** 5x5 Grid, Pickup/Dropoff logic.
- **Algorithm:** Q-Learning (Epsilon-Greedy).
- **Result:** Agent solved the task in 12 steps with a score of 9.
- **Code:** `rl_agent/train_agent.py`

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt