#!/bin/bash
# Remove old Ray session directories from /tmp/ray
du -sh /tmp/ray/ 2>&1
echo "Sessions found:"
ls /tmp/ray/ray/ 2>&1
echo "Removing..."
rm -rf /tmp/ray/ray/session_*
echo "Done. Free space after:"
df -h /tmp
