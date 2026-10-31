"""Desktop window for the app: starts the Streamlit server silently and shows it in a native window."""
import ctypes, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG = APP_DIR / "logs"
LOG.mkdir(exist_ok=True)
TITLE = "Resume Ranking & Fairness Audit"


def popup(msg: str):
    ctypes.windll.user32.MessageBoxW(0, msg, TITLE, 0x10)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    try:
        import webview
    except ImportError:
        popup("pywebview is not installed. Run Install.bat again.")
        return
    port = free_port()
    py = Path(sys.executable).with_name("python.exe")
    py = str(py if py.exists() else sys.executable)
    out = open(LOG / "server.log", "w")
    server = subprocess.Popen(
        [py, "-m", "streamlit", "run", str(APP_DIR / "app.py"), "--server.headless", "true",
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--browser.gatherUsageStats", "false", "--server.maxUploadSize", "200",
         "--theme.base", "light", "--theme.primaryColor", "#4F46E5", "--theme.backgroundColor", "#F4F6FB",
         "--theme.secondaryBackgroundColor", "#FFFFFF", "--theme.textColor", "#1F2937", "--theme.font", "sans serif"],
        stdout=out, stderr=subprocess.STDOUT, creationflags=0x08000000, cwd=str(APP_DIR.parent))
    url = f"http://127.0.0.1:{port}"
    splash = ("<body style='font-family:Segoe UI;display:flex;align-items:center;justify-content:center;"
              "height:100vh;margin:0;color:#444'><div><h2>Starting...</h2><p>First start can take "
              "a few seconds.</p></div></body>")
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True  # needed for the report download buttons
    except Exception:  # noqa: BLE001
        pass
    win = webview.create_window(TITLE, html=splash, width=1320, height=860, min_size=(900, 600))

    def wait_and_load():
        for _ in range(180):
            if server.poll() is not None:
                popup("The app server stopped. See app\\logs\\server.log")
                win.destroy()
                return
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                win.load_url(url)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        popup("The app took too long to start. See app\\logs\\server.log")
        win.destroy()

    try:
        webview.start(wait_and_load)
    finally:
        server.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        (LOG / "desktop.log").write_text(repr(e))
        popup(f"Could not start: {e}")
