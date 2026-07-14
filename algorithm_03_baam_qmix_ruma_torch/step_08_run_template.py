from __future__ import annotations

import json
from pathlib import Path

from step_01_config_and_protocol import BAAMQMixRUMAConfig
from step_06_trainer import BAAMQMIXRUMATrainerTorch


def write_torch_config_template(path: str | Path, cfg: BAAMQMixRUMAConfig | None = None) -> None:
    cfg = cfg or BAAMQMixRUMAConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = BAAMQMixRUMAConfig()
    _trainer = BAAMQMIXRUMATrainerTorch(cfg)
    print("BAAM-QMIX-RUMA PyTorch trainer has been constructed.")
    print("Connect it to a MF-MLe-MUAV-UWR-POMDP environment before training.")
    print(f"default n_points={cfg.n_points}, n_uavs={cfg.n_uavs}, device={cfg.device}")


if __name__ == "__main__":
    main()
