import functools
import xlwt
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for,jsonify,send_from_directory
)
from werkzeug.security import check_password_hash, generate_password_hash
from autoComapp.db import get_db
from autoComapp.autoCore import get_autocom
bp = Blueprint('editmd', __name__, url_prefix='/')

@bp.route('/start',methods=('GET','POST'))
def beforedit():
    if request.method =='POST':
        docname=request.form['docname']
        modelname=request.form['modelname']
        if docname=='':
            print('need docname != ""')
        else:
            session['modelname']=modelname
            db=get_db()
            session['docname'] = docname
            print('start edit doc %s'%(docname))
            if db.execute(
                'SELECT id FROM docs WHERE docname = ?',(docname,)
            ).fetchone() is None:
                db.execute(
                    'INSERT INTO docs(docname,edittime,wakeuptimes) VALUES (?,?,?)',
                    (docname, 0, 0)
                )
                db.commit()
            return redirect(url_for('editmd.editor'))
    return render_template('startpage.html')

@bp.route('/editor')
def editor():
    docname =session.get('docname')
    modelname=session.get('modelname')
    if docname is None:
        redirect(url_for('editmd.beforedit'))
    else:
        print('now edit doc : %s'%(docname))
        print('now edit doc by model : %s' % (modelname))
    return render_template('testeditormd.html')

@bp.route('/editor/add5s')
def add5s():
    docname=session.get('docname')
    db=get_db()
    time=db.execute(
        'SELECT edittime FROM docs WHERE docname= ?',(docname,)
    ).fetchone()[0]
    time=time+5
    db.execute(
        'UPDATE docs SET edittime =? WHERE docname =?',(time,docname)
    )
    db.commit()
    return jsonify(time)

@bp.route('/editor/getautocomp')
def getautocomp():
    docname=session.get('docname')

    #build result
    content=request.args.get('con')
    modelname=session.get('modelname')
    autocom=get_autocom(modelname)
    coms=autocom.doautocom(content)

    db=get_db()
    #add 1 wakeuptime to doc:docname
    wakeuptimes=db.execute('SELECT wakeuptimes FROM docs WHERE docname=?',(docname,)).fetchone()[0]
    wakeuptimes=wakeuptimes+1
    db.execute('UPDATE docs SET wakeuptimes=? WHERE docname=?',(wakeuptimes,docname))
    print(content)
    result=[]
    for com in coms:
        #compute times
        comwt=db.execute('SELECT times FROM wakeupcom WHERE indoc=? AND com=?',(docname,com)).fetchone()
        if comwt is None:
            comwt=0
            re1=0
        else:
            comwt=comwt[0]
            re1 = comwt / (wakeuptimes - 1)
            re1=round(re1*100)
        comct=db.execute('SELECT times FROM chosedcom WHERE indoc=? AND com=?',(docname,com)).fetchone()
        if comct is None:
            comct=0
            re2=0
        else:
            comct=comct[0]
            re2=comct/comwt
            re2=round(re2*100)
        edittime=db.execute('SELECT edittime FROM chosedcom WHERE indoc=? AND com=?',(docname,com)).fetchone()
        if edittime is None:
            re3='-'
        else:
            edittime=edittime[0]
            re3=round(edittime/comwt)
        edited = db.execute('SELECT edited FROM chosedcom WHERE indoc=? AND com=?', (docname, com)).fetchone()
        if edited is None:
            re4 = '-'
        else:
            edited = edited[0]
            re4 = round(edited / comwt*100)
        re=[]
        re.append(com)
        re.append(re1)
        re.append(re2)
        re.append(re3)
        re.append(re4)
        result.append(re)
        # add wakeupcom
        if comwt==0:
            db.execute('INSERT INTO wakeupcom (indoc,com,times) VALUES (?,?,?)',(docname,com,1))
        else:
            db.execute('UPDATE wakeupcom SET times=? WHERE indoc=? AND com=?',(comwt+1,docname,com))

    print(result)
    db.commit()
    return jsonify(result)

@bp.route('/editor/postchosed')
def postchosed():
    docname=session.get('docname')
    com=request.args.get('con')
    ranking=request.args.get('chosed')
    edittime=request.args.get('time')
    edited=request.args.get('diff')
    #print(com)
    db=get_db()
    if db.execute('SELECT * FROM chosedcom WHERE indoc=? AND com=?',(docname,com)).fetchone() is None:
        db.execute('INSERT INTO chosedcom (indoc,com,edittime,ranking,edited,times) VALUES (?,?,?,?,?,?)',(docname,com,edittime,ranking,edited,1,))
    else:
        re=db.execute('SELECT edittime,ranking,edited,times FROM chosedcom WHERE indoc=? AND com=?',(docname,com)).fetchone()
        edittime=re[0]+int(edittime)
        ranking=re[1]+int(ranking)
        edited=re[2]+int(edited)
        times=re[3]+1
        db.execute('UPDATE chosedcom SET edittime=?,ranking=?,edited=?,times=? WHERE indoc=? AND com=?',(edittime,ranking,edited,times,docname,com))

    db.execute('INSERT INTO onechosedcom (indoc,com,edittime,ranking,edited) VALUES (?,?,?,?,?)',
               (docname, com, edittime, ranking, edited,))
    db.commit()
    return jsonify('post success')

@bp.route('/editor/hey')
def hey():
    username=request.args.get('user')
    db = get_db()
    db.execute(
        'INSERT INTO test (username) VALUES (?)',
        (username,)
    )
    db.commit()
    return jsonify('111')

@bp.route('/editor/stat')
def stat():
    db = get_db()
    docname=session.get('docname')
    print(docname)

    work_book = xlwt.Workbook(encoding='utf-8')
    sheet = work_book.add_sheet(docname)
    doc = db.execute('SELECT edittime,wakeuptimes FROM docs WHERE docname=?', (docname,)).fetchone()
    sheet.write(0, 0, 'docname:')
    sheet.write(0, 1, docname)
    sheet.write(1, 0, 'doc total edit time:')
    sheet.write(1, 1, doc[0])
    sheet.write(2, 0, 'total autocomplete wakeuptimes in doc:')
    sheet.write(2, 1, doc[1])
    sheet.write(4, 0, 'wakeup info:')
    sheet.write(5, 0, 'wakeuped content')
    sheet.write(5, 1, 'wakeuped times')
    sheetrow = 6
    cwakeup = db.execute('SELECT * FROM wakeupcom WHERE indoc=?', (docname,))
    for row in cwakeup:
        sheet.write(sheetrow, 0, row[1])
        sheet.write(sheetrow, 1, row[2])
        sheetrow = sheetrow + 1

    sheetrow = sheetrow + 1
    sheet.write(sheetrow, 0, 'chose info:')
    sheetrow = sheetrow + 1
    sheet.write(sheetrow, 0, 'chosed content')
    sheet.write(sheetrow, 1, 'edited time')
    sheet.write(sheetrow, 2, 'chosed rank')
    sheet.write(sheetrow, 3, 'edited characters')
    sheetrow = sheetrow + 1
    cchosed = db.execute('SELECT * FROM onechosedcom WHERE indoc=?', (docname,))
    for row in cchosed:
        sheet.write(sheetrow, 0, row[1])
        sheet.write(sheetrow, 1, row[2])
        sheet.write(sheetrow, 2, row[3])
        sheet.write(sheetrow, 3, row[4])
        sheetrow = sheetrow + 1

    work_book.save('%s.xls' % (docname))

    return  send_from_directory(r'D:\CNM\NTeat\autoCom' , filename='%s.xls' % (docname) , as_attachment=True)
    # return jsonify(docname)