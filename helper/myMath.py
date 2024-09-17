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