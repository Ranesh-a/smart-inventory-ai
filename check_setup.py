import os
import torch
import joblib
import sqlite3
from src.model import LSTMModel  # Make sure this import matches your file structure

def check_step_2():
    print("🕵️ RUNNING SYSTEM CHECK...\n")
    
    # CHECK 1: Files Exist
    files = {
        "Model Weights": "models/lstm_model.pth",
        "Scaler": "scaler.pkl",
        "Database": "project_logs.db"
    }
    
    all_files_exist = True
    for name, path in files.items():
        if os.path.exists(path):
            print(f"✅ {name} found: {path}")
        else:
            print(f"❌ {name} MISSING! Expected at: {path}")
            all_files_exist = False
            
    if not all_files_exist:
        print("\n🚫 STOP: You cannot proceed. Re-run train.py.")
        return

    # CHECK 2: Load Model
    # CHECK 2: Load Model
    try:
        # Initialize model structure
        model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
        
        # Load the file
        checkpoint = torch.load("models/lstm_model.pth", map_location=torch.device('cpu'))
        
        # FIX: Load ONLY the 'model_state_dict' part
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print("✅ Model loaded successfully (Checkpoint format detected).")
    except Exception as e:
        print(f"❌ Model Load Failed: {e}")
        return

    # CHECK 3: Test Database
    try:
        conn = sqlite3.connect("project_logs.db")
        c = conn.cursor()
        # Check if table 'logs' exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs';")
        if c.fetchone():
            print("✅ Database table 'logs' exists.")
        else:
            print("❌ Database exists but table 'logs' is missing. Run db_manager.py.")
        conn.close()
    except Exception as e:
        print(f"❌ Database Check Failed: {e}")
        return

    print("\n🚀 ALL SYSTEMS GO! You are ready for Step 3 (Streamlit App).")

if __name__ == "__main__":
    check_step_2()