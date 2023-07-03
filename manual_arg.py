# configuration of IDS sensor
input_interface = 'wlp4s0' # interface for capture flows
time_interval_training = 1 # in hour
# configuration mqtt client to mqtt broker 
mqtt_broker_address = '172.17.0.1' 
mqtt_port = 1883
mqtt_topic = 'mata_elang_events'
mqtt_username = 'mataelang'
mqtt_password = 'mataelang'