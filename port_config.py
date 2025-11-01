"""Utilities for resolving the network port ScanBass services should bind to."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

_DEFAULT_PORT_ENVS: Sequence[str] = ("SCANBASS_PORT", "PORT")


class PortResolutionError(RuntimeError):
    """Raised when a bind port could not be resolved from the environment."""


def _coerce_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:  # includes None
        raise PortResolutionError(
            f"Port must be an integer value (received {value!r})."
        ) from exc

    if not (0 < port < 65536):
        raise PortResolutionError(
            f"Port must be between 1 and 65535 (received {port})."
        )
    return port


def resolve_port(
    *,
    env_vars: Iterable[str] | None = None,
    fallback: int | None = None,
) -> int:
    """Resolve the network port to bind to.

    Parameters
    ----------
    env_vars:
        Ordered collection of environment variable names to inspect. The first
        one containing a value wins. Defaults to ("SCANBASS_PORT", "PORT").
    fallback:
        Optional integer to use when no environment variable is set. If ``None``
        and no environment variable contains a value, a :class:`PortResolutionError`
        is raised.
    """

    lookup_order: Sequence[str] = tuple(env_vars or _DEFAULT_PORT_ENVS)
    for name in lookup_order:
        raw_value = os.getenv(name)
        if raw_value:
            return _coerce_port(raw_value)

    if fallback is None:
        names = ", ".join(lookup_order)
        raise PortResolutionError(
            f"None of the environment variables ({names}) are set; cannot resolve port."
        )

    return _coerce_port(str(fallback))


def apply_port_to_env(port: int) -> None:
    """Persist the resolved port back into both SCANBASS_PORT and PORT."""

    value = str(port)
    os.environ["SCANBASS_PORT"] = value
    os.environ["PORT"] = value


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve the ScanBass bind port.")
    parser.add_argument(
        "--env",
        dest="env_vars",
        action="append",
        help="Environment variable name to inspect (can be passed multiple times).",
    )
    parser.add_argument(
        "--default",
        dest="fallback",
        type=int,
        default=None,
        help="Fallback port when none of the environment variables are set.",
    )

    args = parser.parse_args(argv)

    try:
        port = resolve_port(env_vars=args.env_vars, fallback=args.fallback)
    except PortResolutionError as exc:
        parser.error(str(exc))

    print(port, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(_cli())
