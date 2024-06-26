import os

# Create package logger
from .modules.log import create_logger
logger = create_logger()

ROOT = os.path.abspath(os.path.dirname(__file__))
NAME = "Salt.AI AudioViz"
PACKAGE = "SaltAI_AudioViz"
MENU_NAME = "SALT"
SUB_MENU_NAME = "AudioViz"
NODES_DIR = os.path.join(ROOT, 'nodes')
EXTENSION_WEB_DIRS = {}
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Load modules
from SaltAI_AudioViz.modules.node_importer import ModuleLoader
module_timings = {}
module_loader = ModuleLoader(PACKAGE)
module_loader.load_modules(NODES_DIR)

# Mappings
NODE_CLASS_MAPPINGS = module_loader.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = module_loader.NODE_DISPLAY_NAME_MAPPINGS

# Timings and such
logger.info("")
module_loader.report(NAME)
logger.info("")

# Export nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
