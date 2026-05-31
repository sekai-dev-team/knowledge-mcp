#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Build initial index if none exists
if [ ! -f /data/index.db ]; then
    echo "[entrypoint] No index found. Building initial index..."
    python indexer.py --full --vault /vault --db /data/index.db
    echo "[entrypoint] Initial index complete."
fi

# Start Syncthing in background (config on persistent /data volume)
SYNCTHING_HOME=/data/syncthing
if [ ! -f "$SYNCTHING_HOME/config.xml" ]; then
    echo "[entrypoint] Generating Syncthing config..."
    syncthing generate --home="$SYNCTHING_HOME"
    # Configure: disable browser, listen on all interfaces, add vault folder
    python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('$SYNCTHING_HOME/config.xml')
r = tree.getroot()
r.find('.//gui/address').text = '0.0.0.0:8384'
r.find('.//options/startBrowser').text = 'false'
dev_id = r.find('.//device').get('id')
f = ET.SubElement(r, 'folder')
f.set('id', 'yui-vault')
f.set('label', 'Yui Vault')
f.set('path', '/vault')
f.set('type', 'sendreceive')
f.set('rescanIntervalS', '30')
f.set('fsWatcherEnabled', 'true')
fd = ET.SubElement(f, 'device')
fd.set('id', dev_id)
tree.write('$SYNCTHING_HOME/config.xml', encoding='utf-8', xml_declaration=True)
print(f'[entrypoint] Syncthing device ID: {dev_id}')
"
fi
echo "[entrypoint] Starting Syncthing..."
syncthing serve --home="$SYNCTHING_HOME" --no-browser &

# Start file change watcher in background
echo "[entrypoint] Starting change watcher..."
python change_watcher.py --vault /vault --db /data/index.db --git-sync &

# Start MCP server in foreground
echo "[entrypoint] Starting MCP server..."
exec python mcp_server.py --vault /vault --db /data/index.db --host 0.0.0.0 --port 8000
