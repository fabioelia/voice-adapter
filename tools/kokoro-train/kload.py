"""Load Kokoro's model code without the package __init__ (which pulls in the G2P pipeline)."""
import sys, types, os, importlib.util
spec = importlib.util.find_spec('kokoro')
if spec is None:
    raise SystemExit('Install the model code first: pip install --no-deps kokoro==0.9.4 (see README.md)')
pkg = types.ModuleType('kokoro'); pkg.__path__ = list(spec.submodule_search_locations); sys.modules['kokoro'] = pkg
from kokoro.model import KModel
W = os.environ.get('KOKORO_WEIGHTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kokoro-weights'))
def load():
    return KModel(config=f'{W}/config.json', model=f'{W}/kokoro-v1_0.pth').eval()
