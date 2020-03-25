# autoComplete

### 文件结构
autocomplete\
|--autoComapp\
|--instance      //instance文件在初始化数据库时会自动建立

### 需要安装的包
pytorch\
seaborn\
flask\
xlwt\
tensorflow\
regex\
toposort

### 使用说明
命令行运行\
windows:
```
set FLASK_APP=autoComapp
flask init-db
flask run
```
其他os试试：
```
export FLASK_APP=autoComapp
flask init-db
flask run
```
运行成功后打开http://127.0.0.1:5000/start \
输入同一文件名称可以持续记录文档编辑时间，进入编辑界面\
crtl+Q可以调用自动补全功能，以当前光标位置到前一个`<.>`或者行首的内容作为待补全内容\
因为僵硬的键盘监听，所以会有一些bug会导致光标移动不正确;)\
为了节省内存，应该在autoCore.py中注释不使用的模型

### 路径修改
需要修改autoComapp\Autocomplete_Transformer下objmain.py和objrun_kg.py中的调用路径和读取文件路径\
/src/objgpt2.py /src/encoder.py中路径 \
因为unpickler中的问题，要修改运行环境下的\Scripts\flask-script.py,增加
```
sys.path.append(r'autoCom')
from autoComapp.objngram import objngram,NGram
```
