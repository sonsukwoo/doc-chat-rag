"""CLI entrypoint for application Postgres bootstrap."""

from __future__ import annotations

import json

from backend.app_db import bootstrap_application_database


def main() -> None:
    result = bootstrap_application_database()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
