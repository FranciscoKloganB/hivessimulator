"""Module with Matlab related classes."""
from __future__ import annotations

import threading
import numpy as np
from typing import Any
from environment_settings import MATLAB_DIR

__engine_available__ = True
try:
    import matlab.engine
except ModuleNotFoundError:
    __engine_available__ = False


class MatlabEngineContainer:
    """Singleton class wrapper containing thread safe access to a MatlabEngine.

    This class provides a thread-safe way to access one singleton
    matlab engine object when running simulations in threaded mode. Having
    one single engine is important, since starting up an engine takes
    approximately 12s (machine dependant), not including the time matlab
    scripts are executing and data convertions between python and matlab and
    back.

    Attributes:
        eng:
            A matlab engine instance, which can be used for example for matrix
            and vector optimization operations throughout the simulations.

            Note:
                MatlabEngine objects are not thread safe, thus it
                is recommended that you utilize  the a wrapper function that
                obtains :py:const:`_LOCK`, before you send any requests to
                ``eng``.
        """
    #: A re-entrant lock used to make `eng` shareable by multiple threads.
    _LOCK = threading.RLock()
    #: A reference to the instance of `MatlabEngineContainer` or `None`.
    _instance: MatlabEngineContainer = None

    @staticmethod
    def get_instance() -> MatlabEngineContainer:
        """Used to obtain a singleton instance of ``MatlabEngineContainer``.

        If one instance already exists that instance is returned,
        otherwise a new one is created and returned.

        Returns:
            A reference to the existing ``MatlabEngineContainer``
            :py:const:`instance <_instance>`.
        """
        if MatlabEngineContainer._instance is None:
            with MatlabEngineContainer._LOCK:
                if MatlabEngineContainer._instance is None:
                    MatlabEngineContainer()
        return MatlabEngineContainer._instance

    def __init__(self) -> None:
        """Instantiates a new MatlabEngineContainer object.

        Note:
            Do not directly invoke constructor, use :py:meth:`get_instance`
            instead.
        """
        if MatlabEngineContainer._instance is None:
            MatlabEngineContainer._instance = self
            if not __engine_available__:
                print("matlab.engine module is not installed.")
                self.eng = None
            else:
                print("loading matlab.engine... this can take a while.")
                try:
                    self.eng = matlab.engine.start_matlab()
                    self.eng.cd(MATLAB_DIR)
                except matlab.engine.EngineError:
                    self.eng = None
        else:
            raise RuntimeError("MatlabEngineContainer is a Singleton. Use "
                               "MatlabEngineContainer.getInstance() to get a "
                               "reference to a MatlabEngineContainer object.")

    # noinspection PyIncorrectDocstring
    def matrix_global_opt(self, a: np.ndarray, v_: np.ndarray) -> Any:
        """Constructs an optimized transition matrix using the matlab engine.

        Constructs an optimized transition matrix using linear programming
        relaxations and convex envelope approximations for the specified steady
        state ``v_``, this is done by invoke the matlabscript
        *matrixGlobalOpt.m* located inside
        :py:const:`~app.environment_settings.MATLAB_DIR`.

        Args:
            a:
                A non-optimized symmetric adjency matrix.
            `v_`:
                A stochastic steady state distribution vector.

        Returns:
            Markov Matrix with ``v_`` as steady state distribution and the
            respective :py:func:`mixing rate
            <app.domain.helpers.matrices.get_mixing_rate>` or ``None``.

        Raises:
            EngineError:
                If you do not have a valid MatLab license.
        """
        if not __engine_available__:
            return None

        with MatlabEngineContainer._LOCK:
            ma = matlab.double(a.tolist())
            mv_ = matlab.double(v_.tolist())
            return self.eng.matrixGlobalOpt(ma, mv_, nargout=1)
