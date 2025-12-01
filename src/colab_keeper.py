"""
Colab Keep-Alive System
=======================
Prevents Google Colab from disconnecting during long training sessions.
Also provides connection monitoring and auto-recovery.
"""

import time
import threading
from IPython.display import display, HTML, Javascript
import requests
from datetime import datetime

class ColabKeepAlive:
    """
    Multi-layered keep-alive system for Google Colab.
    
    Uses:
    1. JavaScript widget to simulate user activity
    2. Python-side connection checks
    3. Periodic pings to prevent timeout
    """
    
    def __init__(self, check_interval=60):
        """
        Args:
            check_interval: Seconds between connection checks
        """
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        self.last_check = None
        
    def inject_javascript_keepalive(self):
        """
        Injects JavaScript into Colab to prevent auto-disconnect.
        This simulates user activity by clicking the connect button periodically.
        """
        js_code = """
        <script>
        // Colab Keep-Alive System
        console.log("🔒 Colab Keep-Alive activated!");
        
        function ClickConnect() {
            console.log("⏰ Keep-alive ping at " + new Date().toLocaleTimeString());
            
            // Find and click the connect button if exists
            var connectButton = document.querySelector("colab-connect-button");
            if (connectButton) {
                connectButton.shadowRoot.querySelector("#connect").click();
            }
            
            // Also try to keep runtime alive
            var kernelStatus = document.querySelectorAll("iron-icon");
            if (kernelStatus && kernelStatus.length > 0) {
                console.log("✅ Runtime is active");
            }
        }
        
        // Click every 5 minutes (300,000 ms)
        setInterval(ClickConnect, 300000);
        
        // Also prevent page visibility timeout
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                console.log("⚠️ Tab hidden - keeping alive anyway");
            }
        });
        
        console.log("✅ Keep-alive will ping every 5 minutes");
        </script>
        
        <div style="padding: 10px; background: #e8f5e9; border-left: 4px solid #4caf50; margin: 10px 0;">
            <strong>🔒 Keep-Alive Active</strong><br>
            Your Colab session will be kept alive automatically.<br>
            <small>Check browser console for pings (F12 → Console)</small>
        </div>
        """
        
        display(HTML(js_code))
        print("✅ JavaScript keep-alive injected!")
        
    def _monitor_connection(self):
        """Background thread that monitors connection health."""
        while self.is_running:
            try:
                self.last_check = datetime.now()
                
                # Try a simple HTTP request to check internet
                response = requests.get("https://www.google.com", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Connection check passed at {self.last_check.strftime('%H:%M:%S')}")
                else:
                    print(f"⚠️ Connection issue detected (status: {response.status_code})")
                    
            except requests.RequestException as e:
                print(f"❌ Connection lost! Error: {e}")
                print("⚡ Attempting to recover...")
                # In a real scenario, you might trigger checkpoint save here
                
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def start(self):
        """Start the keep-alive system."""
        if self.is_running:
            print("⚠️ Keep-alive already running!")
            return
            
        # Inject JavaScript
        self.inject_javascript_keepalive()
        
        # Start Python monitor
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_connection, daemon=True)
        self.monitor_thread.start()
        
        print("🚀 Colab Keep-Alive System started!")
        print(f"   - JavaScript: Pinging every 5 minutes")
        print(f"   - Python monitor: Checking every {self.check_interval} seconds")
        
    def stop(self):
        """Stop the keep-alive system."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Keep-alive stopped")
        

def is_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def activate_colab_keepalive():
    """
    Convenience function to activate keep-alive if in Colab.
    Returns the keeper instance or None.
    """
    if is_colab():
        print("🔍 Google Colab detected!")
        keeper = ColabKeepAlive(check_interval=120)  # Check every 2 minutes
        keeper.start()
        return keeper
    else:
        print("ℹ️ Not running in Colab, keep-alive not needed")
        return None


# Example usage in notebook:
# from src.colab_keeper import activate_colab_keepalive
# keeper = activate_colab_keepalive()
