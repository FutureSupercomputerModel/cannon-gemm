def ceildiv(a, b):
    return -(a // -b)

import math

def bytes2str(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def str2bytes(size_string):
    if size_string[-1] == 'B':
        size_string = size_string[:-1]
    size = float(size_string[:-1])
    if 'K' in size_string:
        size *= 1024
    elif 'M' in size_string:
        size *= 1024*1024
    elif 'G' in size_string:
        size *= 1024*1024*1024
    elif 'T' in size_string:
        size *= 1024*1024*1024*1024
    return size

def str2GBps(bw_string):
    if 'PBps' in bw_string:
        bw = float(bw_string[:-4]) * 1024 * 1024
    elif 'TBps' in bw_string:
        bw = float(bw_string[:-4]) * 1024
    elif 'GBps' in bw_string:
        bw = float(bw_string[:-4])
    elif 'TBps' in bw_string:
        bw = float(bw_string[:-4]) * 1024
    elif 'MBps' in bw_string:
        bw = float(bw_string[:-4]) / 1024
    elif 'KBps' in bw_string:
        bw = float(bw_string[:-4]) / (1024*1024)
    elif 'Bps' in bw_string:
        bw = float(bw_string[:-3]) / (1024*1024*1024)
    return bw
def GBps2str(bw):
    if bw < 0.000001:
        return f"{bw*1024*1024*1024}Bps"
    elif bw < 0.001:
        return f"{bw*1024*1024}KBps"
    elif bw < 1:
        return f"{bw*1024}MBps"
    elif bw < 1024:
        return f"{bw}GBps"
    elif bw < 1024*1024:
        return f"{bw/1024}TBps"
    elif bw < 1024*1024*1024:
        return f"{bw/1024/1024}PBps"
    else:
        return f"{bw/1024/1024/1024}PBps"

def str2energy(energy_string):
    if 'fJ' in energy_string:
        energy = float(energy_string[:-2]) / 1000000
    elif 'pJ' in energy_string:
        energy = float(energy_string[:-2]) / 1000
    elif 'nJ' in energy_string:
        energy = float(energy_string[:-2])
    elif 'uJ' in energy_string:
        energy = float(energy_string[:-2]) * 1000
    elif 'mJ' in energy_string:
        energy = float(energy_string[:-2]) * 1000000
    elif 'J' in energy_string:
        energy = float(energy_string[:-1]) * 1000000000
    return energy

def energy2str(energy):
    if energy < 1e-3:
        return f"{energy*1000000}fJ"
    elif energy < 1:
        return f"{energy*1000}pJ"
    elif energy < 1000:
        return f"{energy}nJ"
    elif energy < 1000000:
        return f"{energy/1000}uJ"
    elif energy < 1000000000:
        return f"{energy/1000000}mJ"
    else:
        return f"{energy/1000000000}J"