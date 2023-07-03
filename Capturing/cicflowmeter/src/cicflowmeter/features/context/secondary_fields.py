#!/usr/bin/env python


from .packet_direction import PacketDirection
import re

def get_secondary_fields(packet, direction) -> tuple:
    """Creates a key signature for a packet.

    Summary:
        Creates a key signature for a packet so it can be
        assigned to a flow.

    Args:
        packet: A network packet
        direction: The direction of a packet

    Returns:
        A tuple of the String IPv4 addresses of the destination,
        the source port as an int,
        the time to live value,
        the window size, and
        TCP flags.

    """

    if "TCP" in packet:
        protocol = "TCP"
    elif "UDP" in packet:
        protocol = "UDP"
    else:
        return '', '', '', '', '',''

    if direction == PacketDirection.FORWARD:
        ip_id, tos, vlan, ttl, b64_data, mpls = load_secondary_fields(packet,protocol)
    else:
        ip_id, tos, vlan, ttl, b64_data, mpls = load_secondary_fields(packet,protocol)

    return ip_id, tos, vlan, ttl, b64_data, mpls


def load_secondary_fields(packet, protocol):
    ip_id = (lambda x: x if x!=None or x!=''  else '' )(packet.id)
    tos = (lambda x: x if x!=None or x!='' else '' )(packet.tos)
    vlan = (lambda packet: packet["Dot1Q"].vlan if "Dot1Q" in packet else 0)(packet)
    ttl = (lambda x: x if x!=None or x!='' else '' )(packet.ttl)
    b64_data = " ".join(find_base64_segments(bytes(packet[protocol].payload).decode('latin-1')))
    mpls = (lambda x: 1 if 'mpls' in x else 0)(packet)
    return ip_id, tos, vlan, ttl, b64_data, mpls
    

def find_base64_segments(payload):
    base64_pattern = r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{4})" 
    base64_segments = re.findall(base64_pattern, payload)
    return base64_segments