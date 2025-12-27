#!/usr/bin/env python3
"""
Test script for GPT-4 powered Healthcare AI API
"""
import requests
import json

API_URL = "http://localhost:3333"

def test_general_query():
    """Test general medical query"""
    print("\n" + "="*60)
    print("TEST 1: General Medical Query")
    print("="*60)
    
    response = requests.post(
        f"{API_URL}/ask",
        json={"prompt": "What is atrial fibrillation?"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_classification():
    """Test signal classification with GPT-4 interpretation"""
    print("\n" + "="*60)
    print("TEST 2: Signal Classification")
    print("="*60)
    
    # Mock ECG signal data
    signal = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.16, 0.14, 0.17, 0.21]
    
    response = requests.post(
        f"{API_URL}/classify",
        json={
            "signal": signal,
            "signal_type": "ECG",
            "clinical_context": "Patient with palpitations"
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

def test_contextual_query():
    """Test query with classification context"""
    print("\n" + "="*60)
    print("TEST 3: Contextual Query")
    print("="*60)
    
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "prompt": "What are the treatment options?",
            "classification": {
                "prediction": "Atrial Fibrillation",
                "confidence": 0.92
            }
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("🧪 Testing GPT-4 Powered Healthcare AI API")
    print("Make sure the server is running at http://localhost:3333")
    
    try:
        test_general_query()
        test_classification()
        test_contextual_query()
        print("\n✅ All tests completed!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure to start the server with: python3 api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
