# edge_node/hardware_publisher.py
import serial
import time
import json
import random
import paho.mqtt.client as mqtt
import platform
import os

# 1. Initialize local MQTT
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# Graceful MQTT Connection Retry Loop
broker_host = os.getenv("MQTT_HOST", "localhost")
while True:
    try:
        print(f"| EDGE | Attempting to connect to Mosquitto Broker at {broker_host}...")
        mqtt_client.connect(broker_host, 1883, 60)
        print(f"| EDGE | SUCCESS: Connected to Mosquitto Broker at {broker_host}.")
        break # Exit the retry loop once connected successfully
    except Exception as e:
        print(f"| EDGE ERROR | Broker connection failed ({e}). Retrying in 3 seconds...")
        time.sleep(3)

# Detect Operating System
port = "COM3" if platform.system() == "Windows" else "/dev/ttyACM0"
baud = 115200
ser = None

telemetry_state = {
    "tps": 0.0, "brake_pressure": 0.0, "speed": 0.0, "gear": 0,
    "rpm": 0, "oil_pressure": 0.0, "fuel_pressure": 0.0, "oil_temp": 0.0,
    "fuel_temp": 0.0, "coolant_temp": 0.0, "lambda": 0.0, "fan_speed": 0.0,
    "battery_voltage": 0.0, "longitude": 0.0, "latitude": 0.0
}

key_mapping = {
    "throttle_pos": "tps",
    "vehicle_speed": "speed",
    "engine_rpm": "rpm",
    "clt_temp": "coolant_temp",
    "bat_voltage": "battery_voltage",
    "oil_pressure": "oil_pressure",
    "air_temp": "fuel_temp",
    "current_gear": "gear",
    "target_lambda": "lambda"
}

last_publish_time = time.time()
last_heartbeat_time = time.time()

mqtt_client.loop_start()

# Robust Hardware Loop (No Mock Data)
while True:
    try:
        # If port is closed or crashed, try to open it
        if ser is None or not ser.is_open:
            print(f"| EDGE | Attempting to connect to {port}...")
            ser = serial.Serial(port, baud, timeout=1)
            print(f"| EDGE | SUCCESS: Connected to Hardware Antenna!")
            ser.reset_input_buffer()
        
        current_time = time.time()
        
        # Send an explicit heartbeat status message every 1 second
        if current_time - last_heartbeat_time > 1.0:
            mqtt_client.publish("bcu-racing/telemetry/status", json.dumps({"antenna": "ONLINE"}))
            print("| EDGE | Heartbeat broadcasted to MQTT!")
            last_heartbeat_time = current_time
            
        # Read the real hardware stream
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode("utf-8", errors="replace").strip()

            print(f"| EDGE | RX: {raw_line}")
            
            if raw_line.startswith("Captured:"):
                try:
                    json_str = raw_line.replace("Captured: ", "").strip()
                    incoming_data = json.loads(json_str)
                    
                    for hw_key, val_str in incoming_data.items():
                        if val_str == "": continue
                        if hw_key in key_mapping:
                            frontend_key = key_mapping[hw_key]
                            telemetry_state[frontend_key] = float(val_str)
                            
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Publish telemetry data stream frames at 10Hz
            if current_time - last_publish_time > 0.1:
                packet = telemetry_state.copy()
                packet["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                mqtt_client.publish("bcu-racing/telemetry/live", json.dumps(packet))
                last_publish_time = current_time
        else:
            time.sleep(0.005) # Prevent 100% CPU usage while waiting for data

    except serial.SerialException:
        print(f"| EDGE WARNING | Hardware connection lost. Retrying...")
        mqtt_client.publish("bcu-racing/telemetry/status", json.dumps({"antenna": "OFFLINE"}))
        if ser: {
            ser.close()
        }
        ser = None
        time.sleep(2) # Wait before attempting to reconnect
    except ValueError:
        # Skips garbled/corrupted serial lines caused by radio interference
        pass
    except Exception as e:
        print(f"| EDGE ERROR | Unexpected fault: {e}")
        time.sleep(1)