
import os, sys, socket, subprocess, time, webbrowser, pathlib, signal

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_REL = "coin_valuer/launch.py"
APP = ROOT / APP_REL
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS / "run.log"

def pick_free_port(start=8501, end=8520):
    for p in range(start, end+1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return None

def wait_for_port(port, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.25)
    return False

def main():
    py = sys.executable
    port_env = os.environ.get("PORT")
    if port_env:
        try:
            port = int(port_env)
        except ValueError:
            port = None
    else:
        port = None
    if port is None:
        port = pick_free_port() or 8501

    # Streamlit config via CLI flags
    args = [
        py, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")

    with open(LOG_FILE, "wb") as logf:
        proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)

    url = f"http://localhost:{port}"
    if wait_for_port(port, timeout=90.0):
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print("Coin Valuer is running at", url)
        print("Logs:", LOG_FILE)
        # Keep waiting until process exits (if ever)
        try:
            proc.wait()
        except KeyboardInterrupt:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
    else:
        print("Timed out waiting for server to start on port", port)
        print("Please check logs at:", LOG_FILE)
        try:
            proc.terminate()
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
