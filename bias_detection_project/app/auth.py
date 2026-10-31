"""Basic accounts: SQLite + salted scrypt (stdlib only). Fine for a demo; not hardened for production."""
import hashlib, hmac, os, re, sqlite3
from pathlib import Path

DB = Path(__file__).parent / "users.db"


def _conn():
    return sqlite3.connect(DB)


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY, email TEXT, salt BLOB, pw_hash BLOB,
            created TEXT DEFAULT CURRENT_TIMESTAMP)""")


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def create_account(username: str, email: str, password: str, confirm: str):
    """Returns (ok, message)."""
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,32}", username):
        return False, "Username must be 3-32 characters: letters, digits, _ . -"
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    salt = os.urandom(16)
    try:
        with _conn() as c:
            c.execute("INSERT INTO users(username,email,salt,pw_hash) VALUES(?,?,?,?)",
                      (username, email.strip(), salt, _hash(password, salt)))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def verify(username: str, password: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT salt, pw_hash FROM users WHERE username=?",
                        (username.strip().lower(),)).fetchone()
    if not row:
        _hash(password, b"\0" * 16)  # keep timing similar for unknown users
        return False
    return hmac.compare_digest(_hash(password, row[0]), row[1])
