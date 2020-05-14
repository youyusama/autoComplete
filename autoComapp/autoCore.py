from flask import g
import sys
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\Autocomplete_Transformer')
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\src')
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp')
from autoComapp.Autocomplete_Transformer.objrun_kg import objrun_kg
#from autoComapp.Autocomplete_Transformer.objmain import objmain
# from autoComapp.src.objgpt2 import objgpt2
# from autoComapp.objngram import objngram,NGram


def get_autocom(modelname):
    # if modelname == 'ngram':
    #     if 'ngram' not in g:
    #         g.ngram = objngram()
    #     return g.ngram
    # if modelname == 'gpt2':
    #     if 'gpt2' not in g:
    #         g.gpt2=objgpt2()
    #     return g.gpt2
    # elif modelname == 'transformer':
    #     if 'transformer' not in g:
    #         g.transformer=objmain()
    #     return g.transformer
    if modelname == 'transformerkg':
        if 'transformerkg' not in g:
            g.transformerkg=objrun_kg()
        return g.transformerkg

# if __name__ == "__main__":
#     obj = objngram()
#     obj.doautocom(
#         'This document defines the ATP software requirements assigned from the onboard train control subsystem')