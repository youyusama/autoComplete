from flask import g
from autoComapp.Autocomplete_Transformer.objmain import objmain

def get_autocom():
    if 'autocom' not in g:
        g.autocom=objmain()
    return g.autocom