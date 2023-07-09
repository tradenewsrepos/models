#!/bin/sh
uvicorn flask_server:app --reload --host 0.0.0.0 --port 8989 --backlog 8 --limit-concurrency 8
