# SPDX-License-Identifier: MIT OR Apache-2.0
"""gitmind CLI:  python -m gitmind <snapshot|backup|report|rollback> [sha]"""
import asyncio, json, sys
from .core import GitMind

def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "report"
    gm = GitMind()
    if not gm.is_repo():
        print("not a git repo (run inside a git working tree)"); sys.exit(1)
    if cmd == "snapshot":
        print(json.dumps(gm.snapshot(), indent=2))
    elif cmd == "backup":
        print(json.dumps(asyncio.run(gm.backup()), indent=2))
    elif cmd == "rollback":
        if len(args) < 2:
            print("usage: python -m gitmind rollback <sha> [--hard]"); sys.exit(2)
        print(json.dumps(gm.rollback_to(args[1], hard="--hard" in args), indent=2))
    else:
        print(json.dumps(gm.report(), indent=2))

if __name__ == "__main__":
    main()
