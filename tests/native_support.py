"""Run frozen-SA integration tests in a clean interpreter, not a cached native."""
from functools import wraps
import inspect
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "build/linux/sa-wall-clock-v1"
CHILD = "LNS2_ISOLATED_SA_TEST"


def isolated_sa_native(test):
    @wraps(test)
    def run(self):
        module = test.__module__
        if module == '__main__':
            module = '.'.join(Path(inspect.getfile(test)).resolve().relative_to(ROOT).with_suffix('').parts)
        identity = f'{module}.{type(self).__qualname__}.{test.__name__}'
        if os.environ.get(CHILD) == identity:
            return test(self)
        if sys.platform != "linux" or not list(NATIVE.glob("lns2_env*.so")):
            self.skipTest("existing frozen SA Linux native required")
        env = dict(os.environ, **{CHILD: identity, "PYTHONPATH": os.pathsep.join((str(NATIVE), str(ROOT)))})
        # Preload before test imports can register another build directory.
        code = "import lns2_env,sys,unittest; unittest.main(module=None,argv=['unittest',sys.argv[1]])"
        result = subprocess.run([sys.executable, "-c", code, identity], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return run
