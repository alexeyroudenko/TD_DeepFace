#!/usr/bin/env python3
"""
Test script to check if all required model weights are available.
"""

import os

def check_weights():
    """Check if all required model weights are present."""
    weights_dir = "/root/.deepface/weights"
    
    # Model weight files that should be present
    required_weights = [
        "age_model_weights.h5",
        "gender_model_weights.h5", 
        "facial_expression_model_weights.h5"
    ]
    
    print(f"Checking weights in directory: {weights_dir}")
    print("-" * 50)
    
    if not os.path.exists(weights_dir):
        print(f"❌ Weights directory does not exist: {weights_dir}")
        return False
    
    all_present = True
    for weight_file in required_weights:
        weight_path = os.path.join(weights_dir, weight_file)
        if os.path.exists(weight_path):
            file_size = os.path.getsize(weight_path)
            print(f"✅ {weight_file} - {file_size:,} bytes")
        else:
            print(f"❌ {weight_file} - MISSING")
            all_present = False
    
    print("-" * 50)
    if all_present:
        print("✅ All required model weights are present!")
    else:
        print("❌ Some model weights are missing!")
    
    return all_present

if __name__ == "__main__":
    check_weights()
