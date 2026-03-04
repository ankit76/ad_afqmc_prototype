import numpy as np

from .. import driver, config
from ..core.system import System
from ..ham.chol import HamChol
from ..prop.afqmc import make_prop_ops
from ..prop.blocks import block
from ..prop.types import QmcParams
from ..prep.pyscf_interface import get_integrals, get_gcisd, get_trial_coeff
from ..trial.gcisd import GcisdTrial, make_gcisd_trial_ops
from ..meas.gcisd import make_gcisd_meas_ops


class Gcisd:
    def __init__(self, cc):
        config.setup_jax()

        mf = cc._scf
        mol = cc.mol
        h0, h1, chol = get_integrals(mf, 1e-12)
        ci1, ci2 = get_gcisd(cc)

        c = get_trial_coeff(mf)

        sys = System(norb=mol.nao, nelec=mol.nelec, walker_kind="generalized")
        ham_data = HamChol(h0, h1, chol, basis="generalized")
        self.trial_data = GcisdTrial(mo_coeff=c, c1=ci1, c2=ci2)
        self.trial_ops = make_gcisd_trial_ops(sys=sys)
        self.meas_ops = make_gcisd_meas_ops(sys=sys)
        self.prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
        self.params = QmcParams(n_eql_blocks=20, n_blocks=200, seed=np.random.randint(0, int(1e6)))
        self.block_fn = block
        self.sys = sys
        self.ham_data = ham_data

    def kernel(self):
        return driver.run_qmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
        )
