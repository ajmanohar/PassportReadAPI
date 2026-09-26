import sys
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Checks service health."""
    print("Testing /health endpoint...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(resp.json())

def test_extraction(endpoint: str, file_path: str):
    """
    Sends a test image to either /extract/visa or /extract/stamp.
    """
    url = f"{BASE_URL}{endpoint}"
    print(f"\nSending {file_path} to {url}...")
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "image/jpeg")}
        response = requests.post(url, files=files)
        
    print(f"Response Code: {response.status_code}")
    try:
        print(response.json())
    except Exception:
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_client.py <visa|stamp> <path_to_image.jpg>")
        print("Example: python test_client.py stamp sample_stamp.jpg")
        test_health()
        sys.exit(0)

    mode = sys.argv[1].lower()
    img_path = sys.argv[2]

    if mode == "visa":
        test_extraction("/extract/visa", img_path)
    elif mode == "stamp":
        test_extraction("/extract/stamp", img_path)
    else:
        print(f"Unknown mode '{mode}'. Choose 'visa' or 'stamp'.")