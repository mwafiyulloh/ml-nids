from manual_arg import *
import os

command = f'cicflowmeter -i {input_interface} -c ./temp_flows/ -t'
os.system(command)