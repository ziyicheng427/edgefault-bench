#!/usr/bin/env bash
# Run TLC model checking on the experiment-lifecycle specification (T09).
# Requires Java 11+ (set JAVA_BIN or JAVA_HOME to override); downloads the
# TLA+ tools jar when absent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TLA_DIR="$ROOT/formal"
JAR="$ROOT/tools/tla2tools.jar"
TLA_VERSION="1.8.0"

if [ -n "${JAVA_BIN:-}" ] && [ -x "${JAVA_BIN:-}" ]; then
  JAVA="$JAVA_BIN"
elif [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
  JAVA="$JAVA_HOME/bin/java"
elif [ -x /opt/homebrew/opt/openjdk/bin/java ]; then
  JAVA=/opt/homebrew/opt/openjdk/bin/java
elif command -v java >/dev/null 2>&1; then
  JAVA=java
else
  echo "error: java (11+) is required to run TLC" >&2
  exit 1
fi

if ! "$JAVA" -version 2>&1 | head -1 | grep -Eq 'version "(1[1-9]|[2-9][0-9])'; then
  echo "error: TLC requires Java 11+, found:" >&2
  "$JAVA" -version >&2 2>&1 || true
  exit 1
fi

if [ ! -f "$JAR" ]; then
  mkdir -p "$ROOT/tools"
  curl -fSL -o "$JAR" \
    "https://github.com/tlaplus/tlaplus/releases/download/v${TLA_VERSION}/tla2tools.jar"
fi

cd "$TLA_DIR"
exec "$JAVA" -jar "$JAR" ExperimentLifecycle
