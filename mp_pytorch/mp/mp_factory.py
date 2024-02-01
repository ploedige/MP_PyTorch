import torch

from mp_pytorch.basis_gn import NormalizedRBFBasisGenerator
from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.basis_gn import ZeroPaddingNormalizedRBFBasisGenerator
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator
from mp_pytorch.phase_gn import LinearPhaseGenerator
from .dmp import DMP
from .prodmp import ProDMP
from .promp import ProMP


class MPFactory:
    @staticmethod
    def init_mp(mp_type: str,
                num_dof: int = 1,
                num_basis: int = 10,
                tau: float = 3,
                delay: float = 0,
                learn_tau: bool = False,
                learn_delay: bool = False,
                mp_args: dict = None,
                dtype: torch.dtype = torch.float32,
                device: torch.device = "cpu"):
        """
        This is a helper class to initialize MPs,
        You can also directly initialize the MPs without using this class

        Create an MP instance given configuration

        Args:
            mp_type: type of movement primitives
            num_dof: the number of degree of freedoms
            num_basis: number of basis functions
            tau: default length of the trajectory
            delay: default delay before executing the trajectory
            learn_tau: if the length is a learnable parameter
            learn_delay: if the delay is a learnable parameter
            mp_args: arguments to a specific mp, refer each MP class
            dtype: data type of the torch tensor
            device: device of the torch tensor


        Returns:
            MP instance
        """

        if mp_args is None:
            mp_args = {}

        # Backward Compatible API Adjustment:        
        if "num_basis" in mp_args:
            num_basis = mp_args.pop("num_basis")

        # Get phase generator
        if mp_type == "promp":
            phase_gn = LinearPhaseGenerator(tau=tau, delay=delay,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay,
                                            dtype=dtype, device=device)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=num_basis,
                dtype=dtype,
                device=device,
                **mp_args)
            mp = ProMP(basis_gn=basis_gn, num_dof=num_dof, dtype=dtype,
                       device=device, **mp_args)

        elif mp_type == 'zero_padding_promp':
            phase_gn = LinearPhaseGenerator(tau=tau,
                                            learn_tau=learn_tau,
                                            learn_delay=learn_delay,
                                            dtype=dtype, device=device)
            basis_gn = ZeroPaddingNormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=num_basis,
                dtype=dtype,
                device=device,
                **mp_args
            )
            mp = ProMP(basis_gn=basis_gn, num_dof=num_dof, dtype=dtype,
                       device=device, **mp_args)

        elif mp_type == "dmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau, delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_args[
                                                  "alpha_phase"],
                                              dtype=dtype, device=device)
            basis_gn = NormalizedRBFBasisGenerator(
                phase_generator=phase_gn,
                num_basis=num_basis,
                dtype=dtype,
                device=device,
                **mp_args)
            mp = DMP(basis_gn=basis_gn, num_dof=num_dof, dtype=dtype,
                     device=device, **mp_args)
        elif mp_type == "prodmp":
            phase_gn = ExpDecayPhaseGenerator(tau=tau, delay=delay,
                                              learn_tau=learn_tau,
                                              learn_delay=learn_delay,
                                              alpha_phase=mp_args[
                                                  "alpha_phase"],
                                              dtype=dtype, device=device)
            basis_gn = ProDMPBasisGenerator(
                phase_generator=phase_gn,
                num_basis=num_basis,
                dtype=dtype,
                device=device,
                **mp_args)
            mp = ProDMP(basis_gn=basis_gn, num_dof=num_dof, dtype=dtype,
                        device=device, **mp_args)
        else:
            raise NotImplementedError

        return mp
