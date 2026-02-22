import network
import socket
import time
import math
from machine import Pin
from servo import Servo

# -------------------------
# WIFI SETTINGS
# -------------------------
SSID = "YOUR_WIFI"
PASSWORD = "YOUR_PASS"

# -------------------------
# SERVO PINS
# -------------------------
servo_front = Servo(Pin(14))
servo_right = Servo(Pin(27))
servo_back  = Servo(Pin(26))
servo_left  = Servo(Pin(25))

# -------------------------
# CONTROL VARIABLES
# -------------------------
GAIN = 30          # movement strength
BASE_TENSION = 10  # keeps cables slightly tight
STEP_SPEED = 0.15  # smoothing speed

system_enabled = True

# Reference positions (start at 90, but DO NOT move them at boot)
pos = {
    "front": 90,
    "right": 90,
    "back":  90,
    "left":  90
}

target = pos.copy()

# -------------------------
# WIFI CONNECT
# -------------------------
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)

    while not wlan.isconnected():
        time.sleep(1)

    ip = wlan.ifconfig()[0]
    print("Connected. IP:", ip)
    return ip

# -------------------------
# APPLY SERVO OUTPUT
# -------------------------
def apply_positions():
    servo_front.write(pos["front"])
    servo_right.write(pos["right"])
    servo_back.write(pos["back"])
    servo_left.write(pos["left"])

# -------------------------
# SMOOTH MOVE LOOP
# -------------------------
def smooth_update():
    global pos

    if not system_enabled:
        return

    for key in pos:
        error = target[key] - pos[key]
        pos[key] += error * STEP_SPEED

    apply_positions()

# -------------------------
# SET POLAR TILT
# -------------------------
def set_polar(r, theta_deg):
    global target

    r = max(0, min(1, r))
    theta = math.radians(theta_deg)

    tilt_x = r * math.cos(theta)
    tilt_y = r * math.sin(theta)

    front =  BASE_TENSION + GAIN * tilt_y
    back  =  BASE_TENSION - GAIN * tilt_y
    right =  BASE_TENSION + GAIN * tilt_x
    left  =  BASE_TENSION - GAIN * tilt_x

    target["front"] = 90 + front
    target["back"]  = 90 + back
    target["right"] = 90 + right
    target["left"]  = 90 + left

    print("TARGET:", target)

# -------------------------
# SIMPLE WEB PAGE
# -------------------------
def webpage():
    return """<!DOCTYPE html>
<html>
<body>
<h2>Platform Control</h2>
<form action="/polar">
R (0-1): <input name="r"><br>
Theta (deg): <input name="theta"><br>
<input type="submit">
</form>
<a href="/off">OFF</a><br>
<a href="/on">ON</a>
</body>
</html>"""

# -------------------------
# WEB SERVER
# -------------------------
def start_server(ip):
    addr = socket.getaddrinfo(ip, 80)[0][-1]
    s = socket.socket()
    s.bind(addr)
    s.listen(5)

    print("Server running at http://", ip)

    while True:
        conn, addr = s.accept()
        request = conn.recv(1024).decode()

        if "/polar" in request:
            try:
                r = float(request.split("r=")[1].split("&")[0])
                theta = float(request.split("theta=")[1].split(" ")[0])
                set_polar(r, theta)
                response = "OK"
            except:
                response = "ERROR"

        elif "/off" in request:
            global system_enabled
            system_enabled = False
            response = "System OFF"

        elif "/on" in request:
            system_enabled = True
            response = "System ON"

        else:
            response = webpage()

        conn.send("HTTP/1.1 200 OK\nContent-Type: text/html\n\n")
        conn.send(response)
        conn.close()

# -------------------------
# MAIN
# -------------------------
ip = connect_wifi()
apply_positions()

import _thread
_thread.start_new_thread(start_server, (ip,))

while True:
    smooth_update()
    time.sleep(0.02)
