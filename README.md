# GPT (Untrained)

Minimal playground for experimenting with a GPT-style model on a personal laptop.  
The parameters are intentionally small so it can run without heavy hardware.

## Features
- **Data Extractor** – Prepares text data for training  
- **Trainer** – Simple training loop for the model  
- **Chatbot** – Basic interactive interface  
- **Bigram Language Model** – Tiny reference model included  

## Notes
- This is an **untrained** setup — more like a framework or skeleton.  
- Designed for learning and tinkering, not for production.  
- Lightweight enough to run on a local machine.  

## Run
```bash
# clone and enter repo
git clone untrainedGPT
cd untrainedGPT

# run trainer
python train_gpt.py

# run chatbot
python chatbot.py
