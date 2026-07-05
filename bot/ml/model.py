"""Live shadow scorer — loads models/model.joblib and produces P(up) per tick.

SHADOW ONLY: the probability is logged by the recorder and shown on the dashboard,
but it makes no trading decisions. The kill-or-scale rule: it may only be promoted
into the decision path after its recorded probabilities beat the market's ask on
calibration (Brier) over 3-4 weeks of logs/ml/ data.

Fails soft everywhere: no artifact / bad features -> predict() returns None.
"""
import os
import threading
from typing import Any, Dict, Optional

from .data import MODELS_DIR

ARTIFACT_PATH = os.path.join(MODELS_DIR, "model.joblib")


class MlModel:
    def __init__(self):
        self._artifact: Optional[Dict[str, Any]] = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    def _load(self) -> bool:
        try:
            mtime = os.path.getmtime(ARTIFACT_PATH)
        except OSError:
            return False
        if self._artifact is not None and self._mtime == mtime:
            return True
        with self._lock:
            try:
                import joblib
                self._artifact = joblib.load(ARTIFACT_PATH)
                self._mtime = mtime  # retrain drops a new file -> hot-reloaded here
                return True
            except Exception:
                self._artifact = None
                return False

    @property
    def info(self) -> Optional[Dict[str, Any]]:
        if not self._load():
            return None
        a = self._artifact
        return {"kind": a.get("kind"), "trained_at": a.get("trained_at"),
                "oos_brier": a.get("oos_brier")}

    def predict_p_up(self, feats: Dict[str, Any]) -> Optional[float]:
        """Calibrated P(window closes above open) from a feature dict, or None."""
        if not self._load():
            return None
        a = self._artifact
        try:
            row = []
            for name in a["features"]:
                v = feats.get(name)
                if v is None:
                    return None
                row.append(float(v))
            return float(a["model"].predict_proba([row])[0, 1])
        except Exception:
            return None


ml_model = MlModel()
