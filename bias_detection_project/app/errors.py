"""Issue flagging: every problem is logged to app/logs/app.log and shown in the UI."""
import logging, traceback
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_DIR / "app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("bias_app")


def flag(store: list, level: str, where: str, message: str, exc: Exception | None = None):
    """level: 'error' | 'warning' | 'info'. store is st.session_state['issues']."""
    item = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "where": where,
            "message": message, "detail": "".join(traceback.format_exception(exc)) if exc else ""}
    store.append(item)
    getattr(_log, "error" if level == "error" else level if level in ("warning", "info") else "info")(
        "%s: %s %s", where, message, item["detail"])
    return item
