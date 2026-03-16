import subprocess
import time
from pyngrok import ngrok

# Kill existing tunnels
ngrok.kill()

# Start Streamlit
print("🚀 Starting app...")
proc = subprocess.Popen(["streamlit", "run", "app.py", "--logger.level=error"])

# Wait for Streamlit to start
time.sleep(4)

# Create ngrok tunnel
print("🌐 Creating internet tunnel...")
public_url = ngrok.connect(8501)

print("\n" + "="*60)
print("✅ APP IS LIVE ON INTERNET!")
print("="*60)
print(f"\n📱 Share this URL with your phone:")
print(f"   {public_url}")
print(f"\n✅ Works on mobile data!")
print(f"✅ No WiFi needed!")
print(f"✅ Lasts 2 hours")
print("\nPress Ctrl+C to stop\n")

try:
    proc.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    ngrok.kill()
    proc.terminate()
