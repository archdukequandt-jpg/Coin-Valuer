import os, socket, subprocess, sys
from pathlib import Path

def _find_free_port(preferred=None):
    candidates = []
    if preferred and str(preferred).isdigit():
        candidates.append(int(preferred))
    candidates += [8501, 8502, 8503, 8599]
    for p in candidates:
        if _check_free(p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def _check_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False

def _pick_free_port(start=8501, end=8520):
    import socket
    for p in range(int(start), int(end) + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", int(p))) != 0:
                return int(p)
    return int(start)

def main(port=None):
    import os, sys, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app_coin.py")

    # Base env; avoid user site-packages & usage stats
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # Choose a port (prefer provided; else first free)
    if port is None:
        try:
            port = int(os.environ.get("COIN_VALUER_PORT", "8501"))
        except Exception:
            port = 8501
    try:
        auto = os.environ.get("COIN_VALUER_AUTO_PORT", "1")
        if str(auto).strip() != "0":
            port = _pick_free_port(port, port + 19)
    except Exception:
        pass

    # Use space-separated flags for max compatibility with different Streamlit builds
    cmd = [sys.executable, "-m", "streamlit", "run", app_path,
           "--server.port", str(port),
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]

    try:
        # Don't crash the launcher if Streamlit returns a non-zero exit code
        subprocess.run(cmd, env=env, check=False)
    except Exception:
        # Fallback: try a very minimal invocation on another free port
        try:
            port2 = _pick_free_port(port + 1, port + 20)
        except Exception:
            port2 = (port or 8501) + 1
        fallback = [sys.executable, "-m", "streamlit", "run", app_path,
                    "--server.port", str(port2), "--server.headless", "true"]
        try:
            subprocess.run(fallback, env=env, check=False)
        except Exception:
            # Final attempt without any flags at all
            subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], env=env, check=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Port to serve the app on")
    args = parser.parse_args()
    main(port=args.port)