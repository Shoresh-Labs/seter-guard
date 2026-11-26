import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The suite states what is masked with the default settings; a variable left in the shell must not change that.
os.environ.pop("SETER_IL_ID_CHECK", None)
os.environ.pop("SETER_IL_COURT_CASE", None)
os.environ.pop("SETER_ENTITY_PRESET", None)
os.environ.pop("SETER_DECODER", None)
