# autoComplete

### 建议的文件结构
autocomplete\
|--autoComapp\
|--instance      //instance文件在初始化数据库时会自动建立

### 需要安装的包
pytorch\
seaborn\
flask

### 使用说明
命令行运行\
windows:
```
set FLASK_APP=autoComapp
flask init-db
flask run
```
其他os：(我也没试过)
```
export FLASK_APP=autoComapp
flask init-db
flask run
```
运行成功后打开http://127.0.0.1:5000/start\
输入同一文件名称可以持续记录文档编辑时间，进入编辑界面\
crtl+Q可以调用自动补全功能，以当前光标位置到前一个`<.>`或者行首的内容作为待补全内容\
因为僵硬的键盘监听，所以会有一些bug会导致光标移动不正确;)
