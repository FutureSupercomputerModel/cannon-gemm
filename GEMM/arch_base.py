import math
class Arch_base:
    child_arch=None
    buffer_bw=None
    level=None
    total_levels = None
    matrix_block_dim_min=None
    def get_gemm_latency_energy(self, M:int, K:int, N:int):
        return None
    def get_max_gemm_size(self):
        return (None, None, None)
    def debugprint(self, *args):
        print(f"LEVEL {self.level}:",  *args)