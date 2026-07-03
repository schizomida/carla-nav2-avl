# Dinosaur deploy pieces (as-run 2026-07-02)

Operational glue for the car computer ("dinosaur", Jetson AGX Orin). These are
host-specific by design — paths under `/home/dinosaur`, the car's Tailscale IP
in the dashboard — because they document exactly what runs on the vehicle.

| file | what |
|---|---|
| `full_stack_restart.sh` | brings up the whole stack in a tmux session `percept` (costmap uses the repo `config/perception_dinosaur.yaml`; viz runs the repo `tools/viz_node.py`): sensors (TF + velodyne + xsens), the 3 ZED X wrappers (sequential, 40 s apart, each in a self-restarting watchdog loop), the fused costmap node with `config/perception_dinosaur.yaml`, `tools/viz_node.py`, web_video_server (:8080) and the dashboard file server (:8090). tmux runs on a DEDICATED socket -- attach with `tmux -L percept attach` -- inside a systemd user scope so it survives SSH teardown (linger must be enabled: `sudo loginctl enable-linger dinosaur`). |
| `clean_camera_restart.sh` | recovery for wedged GMSL cameras ("CAMERA STREAM FAILED TO START" / Argus timeouts): kills the wrappers, restarts nvargus + zed_x_daemon, brings cameras back sequentially. Restarting the daemon under live wrappers wedges the others — always go through this script. |
| `live_dashboard.html` | the phone/PC live page (fused BEV, live costmap, 3 raw feeds as MJPEG). Deploy to `/home/dinosaur/live_dashboard/index.html`; served on :8090, streams pulled from web_video_server on :8080. |

Known gap: none of this auto-starts on boot — a power cycle kills everything
except the joystick webui (which has its own systemd service). Wrap
`full_stack_restart.sh` in a systemd unit if that keeps biting.
