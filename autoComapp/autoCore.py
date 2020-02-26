from flask import g
from autoComapp.Autocomplete_Transformer.objrun_kg import objrun_kg
from autoComapp.Autocomplete_Transformer.objmain import objmain

def get_autocom():
    if 'autocom' not in g:
        g.autocom=objrun_kg()
    return g.autocom