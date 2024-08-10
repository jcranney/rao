#!/usr/bin/env python
import torch
from astropy.io import fits
import time
import argparse
from pydantic import BaseModel, ConfigDict

parser = argparse.ArgumentParser(
    "run a linear AO simulation"
)
parser.add_argument("--device", nargs="?", default="cpu", type=str)
parser.add_argument("--verbose", "-v", action="count", default=0)
args = parser.parse_args()

device = args.device
verbose = args.verbose


def load(name):
    import numpy as np
    data = fits.open(name)[0].data.astype(np.float32)
    tensor = torch.tensor(data, device=device)
    return tensor


def start_message(msg):
    if verbose:
        print(msg, end="", flush=True)
    return time.time()


def end_message(t):
    if verbose:
        print(f": {time.time()-t:0.1f} sec")


# these are the inputs to the simulation, defining EVERYTHING
t = start_message("loading input matrices")
cmm = load("/tmp/c_meas_meas.fits")
cmp = load("/tmp/c_meas_phi.fits")
ckp = load("/tmp/c_phip1_phi.fits")
cpp = load("/tmp/c_phi_phi.fits")
ctm = load("/tmp/c_ts_meas.fits")
dmc = load("/tmp/d_meas_com.fits")
dtc = load("/tmp/d_ts_com.fits")
dpc = load("/tmp/d_phi_com.fits")
end_message(t)

# these are derived products needed to run the simulation
t = start_message("solving derived matrices")
dkp = torch.linalg.solve_ex(cpp, ckp, left=False)[0]
_cvv = cpp-torch.einsum("ij,jk,lk->il", dkp, cpp, dkp)
cvv_factor = torch.linalg.cholesky_ex(_cvv)[0]
cpp_factor = torch.linalg.cholesky_ex(cpp)[0]
dmp = torch.linalg.solve_ex(cpp, cmp, left=False)[0]
end_message(t)


class AOSystem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cvv_factor: torch.Tensor
    cpp_factor: torch.Tensor
    dkp: torch.Tensor
    dmp: torch.Tensor
    dmc: torch.Tensor
    dpc: torch.Tensor
    _phi: torch.Tensor = None
    _com: torch.Tensor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phi = torch.zeros(self.cpp_factor.shape[0], device=device)
        self._com = torch.zeros(self.dmc.shape[1], device=device)
        self.reset()

    def reset(self):
        self._com[:] = 0.0
        self._phi[:] = 0.0
        self._phi += self._randmult(self.cpp_factor)

    def step(self):
        self._phi[:] = torch.einsum(
            "ij,j->i",
            self.dkp,
            self._phi,
        ) + self._randmult(self.cvv_factor)
        
        meas = torch.einsum(
            "ij,j->i",
            self.dmp,
            self._phi,
        ) + torch.einsum(
            "ij,j->i",
            self.dmc,
            self._com,
        )
        return meas

    def set_command(self, com):
        self._com[:] = com[:]

    def _randmult(self, mat: torch.Tensor):
        return torch.einsum(
            "ij,j->i",
            mat,
            torch.randn([mat.shape[1]], device=device)
        )

    @property
    def phi_atm(self):
        return self._phi.reshape([64, 64])

    @property
    def phi_cor(self):
        return (self.dpc @ self._com).reshape([64, 64])

    @property
    def phi_res(self):
        return self.phi_atm + self.phi_cor

    @property
    def perf(self):
        rms_wfe_rad = self.phi_res.std()
        return f"{torch.exp(-rms_wfe_rad**2):8.3e}, {rms_wfe_rad:0.3f}"


# build AO system
t = start_message("building AO system")
aosys = AOSystem(
    cvv_factor=cvv_factor,
    cpp_factor=cpp_factor,
    dkp=dkp,
    dmp=dmp,
    dmc=dmc,
    dpc=dpc,
)
end_message(t)

# build reconstructor
t = start_message("building reconstructor matrix LHS")
_dcc_reg = 1e-3 * torch.eye(dmc.shape[1], device=device)
_dct = torch.linalg.solve_ex(
    torch.einsum(
        "ji,jk->ik",
        dtc,
        dtc,
    ) + _dcc_reg,
    dtc,
    left=False,
)[0].T
end_message(t)
t = start_message("building reconstructor matrix RHS")
_cmm_reg = 1e-3 * torch.eye(cmm.shape[0], device=device)
_dtm = torch.linalg.solve_ex(cmm + _cmm_reg, ctm, left=False)[0]
dcm = _dct @ _dtm
end_message(t)

# run AO loop:
aosys.reset()
print(aosys.perf)
m = aosys.step()
c = torch.zeros(dcm.shape[0], device=device)
gain = 0.4
t = start_message("running 100 steps")
for _ in range(100):
    c = (1-gain) * c - gain * dcm @ (m - dmc @ c)
    aosys.set_command(c)
    m = aosys.step()
end_message(t)
print(aosys.perf)
