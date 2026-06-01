# edge_node/hardware_publisher.py
import serial
import time
import json
import random
import paho.mqtt.client as mqtt

# 1. Initialize local MQTT
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
try:
    mqtt_client.connect("localhost", 1883, 60)
    print("| EDGE | Connected to Local Mosquitto Broker.")
except Exception as e:
    print(f"| EDGE ERROR | Could not connect to MQTT Broker: {e}")
    exit(1)

port = "/dev/ttyACM0" #what if windows
baud = 115200
ser = None

# Robust Hardware Loop (No Mock Data)
while True:
    try:
        # If port is closed or crashed, try to open it
        if ser is None or not ser.is_open:
            print(f"| EDGE | Attempting to connect to {port}...")
            ser = serial.Serial(port, baud, timeout=1)
            print(f"| EDGE | SUCCESS: Connected to Hardware Antenna!")
            ser.reset_input_buffer()

        # Read the real hardware stream
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode("utf-8", errors="replace").strip()
            parts = raw_line.split(',')
            
            # Ensure we have a complete CSV string before processing
            if len(parts) >= 15:
                packet = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tps": float(parts[0]),
                    "brake_pressure": float(parts[1]),
                    "speed": float(parts[2]),
                    "gear": int(parts[3]),
                    "rpm": int(parts[4]),
                    "oil_pressure": float(parts[5]),
                    "fuel_pressure": float(parts[6]),
                    "oil_temp": float(parts[7]),
                    "fuel_temp": float(parts[8]),
                    "coolant_temp": float(parts[9]),
                    "lambda": float(parts[10]),
                    "fan_speed": float(parts[11]),
                    "battery_voltage": float(parts[12]),
                    "longitude": float(parts[13]),
                    "latitude": float(parts[14])
                }
                # Publish valid data to the network
                mqtt_client.publish("bcu-racing/telemetry/live", json.dumps(packet))
        else:
            time.sleep(0.005) # Prevent 100% CPU usage while waiting for data

    except serial.SerialException:
        print(f"| EDGE WARNING | Hardware connection lost. Waiting for antenna to return...")
        if ser:
            ser.close()
            ser = None
        time.sleep(2) # Wait before attempting to reconnect
    except ValueError:
        # Skips garbled/corrupted serial lines caused by radio interference
        pass
    except Exception as e:
        print(f"| EDGE ERROR | Unexpected fault: {e}")
        time.sleep(1)