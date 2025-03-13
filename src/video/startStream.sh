#!/bin/bash
ssh "foosball@192.168.1.3" "./gstSndr.sh"
ssh "foosball@192.168.1.3" "pkill gst-launch-1.0"