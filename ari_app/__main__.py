"""Run the ARI bot: `python -m ari_app` (foreground asyncio, no HTTP server)."""

from __future__ import annotations

import asyncio
import logging
import sys

from ari_app.ari_loop import run_ari_forever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    try:
        asyncio.run(run_ari_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
