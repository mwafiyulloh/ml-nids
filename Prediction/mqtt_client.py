import paho.mqtt.client as mqtt
import json
from manual_arg import *

class mqtt_client:
    def __init__(self) -> None:
        self.client = mqtt.Client()
        self.client.username_pw_set(username=mqtt_username, password=mqtt_password) 

    def on_connect_mqtt(self):
        if self.client.connect(mqtt_broker_address, mqtt_port, 60) != 0:
            print('Could not connect to MQTT Broker!')
        print('Successfully connect to MQTT Broker!')

    def send_log(self, log):
        data = json.dumps(log)
        self.client.publish(mqtt_topic, data, 0)

    def disconnect_mqtt(self):
        self.client.disconnect()    