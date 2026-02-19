module.exports = {
  apps: [
    {
      name: "whatsapp-bridge",
      script: "dist/index.js",
      cwd: "/Users/joel/Projects/Nanobot/bridge",
      interpreter: "node",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,
    },
    {
      name: "nanobot-api",
      script: "/Users/joel/Projects/Nanobot/start-api.py",
      interpreter: "/Users/joel/Projects/Nanobot/.venv/bin/python",
      cwd: "/Users/joel/Projects/Nanobot",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,
    },
  ],
};
