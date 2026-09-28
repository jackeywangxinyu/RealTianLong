#!/usr/bin/env bash
# ============================================================
#  [INPUT]: 需要 Java 17+ 与可访问 dist.neo4j.org 的网络
#  [OUTPUT]: 在 $NEO4J_HOME（默认 ~/.local/neo4j）下载、配置并启动 Neo4j Community，仅监听 127.0.0.1
#  [POS]: scripts 的本地开发基础设施；无 Docker 环境下为契约测试与持久化游玩提供真实图数据库
#  [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
# ============================================================
set -euo pipefail

VERSION="${NEO4J_VERSION:-5.26.12}"
HOME_DIR="${NEO4J_HOME:-$HOME/.local/neo4j}"
PASSWORD="${NEO4J_PASSWORD:?请先设置 NEO4J_PASSWORD（至少 8 位）}"
DIR="$HOME_DIR/neo4j-community-$VERSION"

if [[ ! -d "$DIR" ]]; then
  mkdir -p "$HOME_DIR"
  curl -fsSL "https://dist.neo4j.org/neo4j-community-$VERSION-unix.tar.gz" | tar xz -C "$HOME_DIR"
  "$DIR/bin/neo4j-admin" dbms set-initial-password "$PASSWORD"
  sed -i 's/^#server.default_listen_address=0.0.0.0/server.default_listen_address=127.0.0.1/' "$DIR/conf/neo4j.conf"
  {
    echo "server.memory.heap.initial_size=512m"
    echo "server.memory.heap.max_size=1g"
    echo "server.memory.pagecache.size=256m"
  } >> "$DIR/conf/neo4j.conf"
fi

"$DIR/bin/neo4j" start
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null http://127.0.0.1:7474; then
    echo "Neo4j 就绪：export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=***"
    exit 0
  fi
  sleep 1
done
echo "Neo4j 启动超时，查看 $DIR/logs/neo4j.log" >&2
exit 1
