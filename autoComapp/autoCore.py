from flask import g
import sys
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\Autocomplete_Transformer')
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\src')
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp')
from autoComapp.Autocomplete_Transformer.objrun_kg import objrun_kg
from autoComapp.Autocomplete_Transformer.objmain import objmain
from autoComapp.src.objgpt2 import objgpt2


def get_autocom(modelname):
    if modelname == 'transformer':
        if 'transformer' not in g:
            g.transformer=objmain()
            return g.transformer
    elif modelname == 'transformerkg':
        if 'transformerkg' not in g:
            g.transformerkg=objrun_kg()
            return g.transformerkg
    elif modelname == 'gpt2':
        if 'gpt2' not in g:
            g.gpt2=objgpt2()
            return g.gpt2
