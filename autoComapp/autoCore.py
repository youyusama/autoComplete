from flask import g
import sys
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\Autocomplete_Transformer')
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp')
from autoComapp.Autocomplete_Transformer.objrun_kg import objrun_kg
from autoComapp.Autocomplete_Transformer.objmain import objmain


def get_autocom(modelname):
    if modelname == 'transformer':
        if 'transformer' not in g:
            g.transformer=objmain()
            return g.transformer
    elif modelname == 'transformerkg':
        if 'transformerkg' not in g:
            g.transformerkg=objrun_kg()
            return g.transformerkg
