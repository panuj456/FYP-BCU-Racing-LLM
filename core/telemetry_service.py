import os
import csv
import json
import time
import shutil
import asyncio
from queue import Queue
import paho.mqtt.client as mqtt

class TelemetryService:
    def __init__(self):
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        

        # Session State tracking
        self.session_active = False
        self.session_name = ""
        self.active_filepath = ""
        self.file = None
        self.writer = None
        self.loop = None
        self.session_start_time = None

        self.antenna_online = False
        self.last_antenna_heartbeat = 0

        # Asyncio Queue that bridges the synchronous MQTT thread to the FastAPI WebSocket loop
        self.queue = asyncio.Queue(maxsize=100)

        # The definitive list of telemetry fields based on your specifications
        self.headers = [
            "Timestamp", "TPS", "Brake_Pressure", "Speed", "Gear", "RPM",
            "Oil_Pressure", "Fuel_Pressure", "Oil_Temp", "Fuel_Temp",
            "Coolant_Temp", "Lambda", "Fan_Speed", "Battery_Voltage",
            "Longitude", "Latitude"
        ]

    def initialize_mqtt(self, loop):
        """Starts the background MQTT network loop and attaches the main thread's async loop."""
        self.loop = loop
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message       
        # Connects to the local mosquitto container over the internal docker bridge network
        self.mqtt_client.connect("mosquitto", 1883, 60)
        self.mqtt_client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print("| BACKEND | Subscribed to telemetry streams from Mosquitto.")
        # Subscribe to BOTH topics
        client.subscribe("bcu-racing/telemetry/live")
        client.subscribe("bcu-racing/telemetry/status")
        
    def _on_message(self, client, userdata, msg):
        """Routes incoming MQTT packets based on their topic."""
        try:
            packet = json.loads(msg.payload.decode())
            
            # 1. Handle Antenna Status Heartbeats
            if msg.topic == "bcu-racing/telemetry/status":
                self.antenna_online = (packet.get("antenna") == "ONLINE")
                self.last_antenna_heartbeat = time.time()
                print(f"| BACKEND | Heartbeat received: {packet}")
                return

            # 2. Handle Live Data Stream
            if msg.topic == "bcu-racing/telemetry/live":
            # A. Send to the live Dashboard UI (always, even if not recording)
                if self.loop:
                    def safe_push():
                        if self.queue.full():
                            try: self.queue.get_nowait()
                            except: pass
                        self.queue.put_nowait(packet)
                    self.loop.call_soon_threadsafe(safe_push)

                # B. Record to CSV (only happens if the user clicked "Start Session")
                if self.session_active and getattr(self, 'writer', None):
                    row = [packet.get(k.lower(), packet.get(k, 0)) for k in self.headers]
                    self.writer.writerow(row)
                    self.file.flush()
                    
        except Exception as e:
            print(f"| MQTT ERROR | Failed processing packet: {e}")
    """
    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # 1. Handle Antenna Status Updates
            if msg.topic == "bcu-racing/telemetry/status":
                telem_service.antenna_online = (payload.get("antenna") == "ONLINE")
                telem_service.last_antenna_heartbeat = time.time()
                return
    
            # 2. Stream Live Metrics regardless of session state
            if msg.topic == "bcu-racing/telemetry/live":
                # Always feed the websocket queue so the dashboard updates live
                asyncio.run_coroutine_threadsafe(telem_service.queue.put(payload), telem_service.loop)
                
                # Only log/write to CSV if the session is officially recording
                if telem_service.session_active:
                    telem_service.save_to_csv(payload)
        except Exception as e:
            print(f"MQTT Callback Error: {e}")
    """

    def start_session(self, session_name):
        os.makedirs("data/active", exist_ok=True)
        os.makedirs("data/outbox", exist_ok=True)
        
        self.session_name = f"{session_name}_{int(time.time())}"
        self.active_filepath = f"data/active/{self.session_name}.csv"
        
        self.file = open(self.active_filepath, mode='w', newline='') 
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        self.file.flush()
        
        self.session_active = True
        self.session_start_time = time.time()
        print(f"| LOG | Session started: {self.session_name}")
        
    def stop_session(self):
        if not self.session_active:
            return

        self.session_active = False
        self.session_start_time = None
        
        if hasattr(self, 'file') and self.file:
            self.file.close()
        
        outbox_filepath = f"data/outbox/{self.session_name}.csv"
        archive_filepath = f"data/archive/{self.session_name}.csv"
        #archive_filepath = os.path.join("data", "archive", f"{self.session_name}.csv")
        
        shutil.move(self.active_filepath, outbox_filepath)
        shutil.copy2(self.active_filepath, archive_filepath)
        print(f"| LOG | Session complete. Moved to Outbox.")

    