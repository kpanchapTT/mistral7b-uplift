#!/bin/bash
# used for debugging resource issues post-crash

# Get current date and time
START_DATETIME=$(date '+%Y-%m-%d-%H_%M_%S')
# Define the log file location
LOG_FILE="resource_logs/${START_DATETIME}.log"
echo "logging resources every 5 seconds to ${LOG_FILE}"

while true; do
    CUR_DATETIME=$(date '+%Y-%m-%d-%H_%M_%S')
    # Get CPU usage using mpstat. The sed command extracts the CPU idle percentage and calculates usage.
    CPU_USAGE=$(mpstat 1 1 | awk '/Average/ {print 100 - $12}')

    # Get RAM usage using free command. The awk command extracts the used and total memory values.
    RAM_USAGE=$(free -g | awk '/Mem/ {print $3"/"$2}')

    # Log the data with timestamp
    echo "$CUR_DATETIME, CPU Usage: $CPU_USAGE%, RAM Usage: $RAM_USAGE GB" >> "${LOG_FILE}"
    sleep 5
done
