#!/bin/bash
# Entrypoint for the paper-trader container.
# Writes a crontab and runs cron in the foreground so Docker can manage it.
#
# Schedule: Mon–Fri at 14:30 UTC = 9:30 AM ET (EST offset; adjust to 13:30 UTC during EDT / summer)
# To override the schedule, set the CRON_SCHEDULE env var in docker-compose.yml.

set -e

CRON_SCHEDULE="${CRON_SCHEDULE:-30 13 * * 1-5}"
LOG_FILE="/app/logs/paper-trader.log"

echo "Paper trader cron schedule: ${CRON_SCHEDULE} (UTC)"
echo "Logs → ${LOG_FILE}"

# Write the crontab for root
cat > /etc/cron.d/paper-trader <<EOF
PATH=/usr/local/bin:/usr/bin:/bin
${CRON_SCHEDULE} root cd /app && uv run python paper_trader.py >> ${LOG_FILE} 2>&1
EOF

chmod 0644 /etc/cron.d/paper-trader
crontab /etc/cron.d/paper-trader

# Tail the log in the background so `docker compose logs` shows output
touch "${LOG_FILE}"
tail -F "${LOG_FILE}" &

echo "Cron started. Waiting for next scheduled run..."
# Run cron in foreground (keeps container alive)
exec cron -f
