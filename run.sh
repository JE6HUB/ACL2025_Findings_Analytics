#!/bin/bash

cd "$(dirname "$0")"

# 仮想環境があれば起動
if [ -d "venv" ]; then
  source venv/bin/activate
else
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
fi

# Streamlitを起動
streamlit run app.py