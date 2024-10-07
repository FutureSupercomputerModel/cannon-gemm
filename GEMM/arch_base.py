import math
class Log:
    def __init__(self):
        self.buffer_access = 0
        self.interconnect_bits = 0
        self.mac = 0
        self.T_prep = 0
        self.T_compute = 0
        self.T_send = 0
        self.T_store = 0

    def update_data_transfer(self, num_iter):
        self.buffer_access *= num_iter
        self.interconnect_bits *= num_iter
        self.mac *= num_iter
    def update_latency(self, num_iter):
        self.T_prep *= num_iter
        self.T_compute *= num_iter
        self.T_send *= num_iter
        self.T_store *= num_iter
    def toString(self):
        return f"T_prep: {self.T_prep}, T_compute: {self.T_compute}, T_send: {self.T_send}, T_store: {self.T_store}, buffer_access: {self.buffer_access}, interconnect_bits: {self.interconnect_bits}, mac: {self.mac}"
    def to_dict(self):
        return {
            "T_prep": self.T_prep,
            "T_compute": self.T_compute,
            "T_send": self.T_send,
            "T_store": self.T_store,
            "buffer_access": self.buffer_access,
            "interconnect_bits": self.interconnect_bits,
            "mac": self.mac
        }
class Arch_base:
    child_arch=None
    buffer_bw=None
    level=None
    total_levels = None
    matrix_block_dim_min=None
    log = None
    def get_gemm_latency_energy(self, M:int, K:int, N:int):
        return None
    def get_max_gemm_size(self):
        return (None, None, None)
    def debugprint(self, *args):
        print(f"LEVEL {self.level}:",  *args)