"""Голосовые эмбеддинги ECAPA-TDNN (speechbrain) — 192-мерный «отпечаток» голоса."""
import logging
import threading

import numpy as np
import torch

from ..config import Settings

log = logging.getLogger(__name__)


class VoiceEmbedder:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                from speechbrain.inference.speaker import EncoderClassifier

                log.info("Загрузка ECAPA-TDNN...")
                self._model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(self._cfg.models_dir / "ecapa"),
                    run_opts={"device": "cpu"},
                )
                log.info("ECAPA-TDNN загружен")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 16 кГц → L2-нормированный вектор (192,)."""
        self.load()
        with torch.no_grad():
            wav = torch.from_numpy(audio).float().unsqueeze(0)
            emb = self._model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
