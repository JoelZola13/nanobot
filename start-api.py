"""PM2 entry point for nanobot API server."""
import os
import sys

# Ensure we're in the right directory
os.chdir("/Users/joel/Projects/Nanobot")
sys.path.insert(0, "/Users/joel/Projects/Nanobot")

from nanobot.api_server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=18790, log_level="info")
