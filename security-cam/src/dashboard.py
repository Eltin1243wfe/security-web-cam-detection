"""
Local web dashboard: live MJPEG feed, armed/disarmed control, schedule,
"It's OK" acknowledge button, and the recent alert log. Runs in the main
thread while detection runs on a background thread (see main.py) —
Flask's dev server is fine at this scale; if this ever needs to handle
real concurrent load, that's a stage 7 problem, not now.

Deliberately bound to localhost only (see main.py's app.run call) —
there's no login on this thing, so it shouldn't be reachable from the
rest of the WiFi network by default. Opening it up to the LAN is a
one-line config change later, but not a silent default.
"""

import time

from flask import Flask, Response, jsonify, request, render_template, send_from_directory

import config


def create_app(state):
    app = Flask(__name__, template_folder="../templates")

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                jpeg = state.get_latest_jpeg()
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                # Caps the stream's own refresh rate independent of how
                # fast detection is actually running — no reason to push
                # frames to the browser faster than ~20fps looks like.
                time.sleep(0.05)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        use_time_window, start_hour, end_hour = state.get_schedule()
        return jsonify({
            "armed_override": state.is_armed_override_enabled(),
            "use_time_window": use_time_window,
            "armed_start_hour": start_hour,
            "armed_end_hour": end_hour,
            "acknowledged": state.is_acknowledged(),
            "alert_log": state.get_alert_log(),
        })

    @app.route("/toggle_armed", methods=["POST"])
    def toggle_armed():
        state.set_armed_override(not state.is_armed_override_enabled())
        return jsonify({"armed_override": state.is_armed_override_enabled()})

    @app.route("/acknowledge", methods=["POST"])
    def acknowledge():
        state.acknowledge()
        return jsonify({"acknowledged": True})

    @app.route("/schedule", methods=["POST"])
    def update_schedule():
        data = request.get_json()
        state.set_schedule(
            use_time_window=bool(data.get("use_time_window")),
            start_hour=int(data.get("armed_start_hour")),
            end_hour=int(data.get("armed_end_hour")),
        )
        return jsonify({"ok": True})

    @app.route("/snapshots/<path:filename>")
    def snapshots(filename):
        return send_from_directory(config.SNAPSHOT_DIR, filename)

    return app