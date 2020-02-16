import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for,jsonify
)
from werkzeug.security import check_password_hash, generate_password_hash
from autoComapp.db import get_db
from autoComapp.autoCore import get_autocom
bp = Blueprint('editmd', __name__, url_prefix='/')

@bp.route('/start',methods=('GET','POST'))
def beforedit():
    if request.method =='POST':
        docname=request.form['docname']
        if docname=='':
            print('need docname != ""')
        else:
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
    if docname is None:
        redirect(url_for('editmd.beforedit'))
    else:
        print('now edit doc %s'%(docname))
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
    db=get_db()
    #add 1 wakeuptime to doc:docname
    wakeuptimes=db.execute('SELECT wakeuptimes FROM docs WHERE docname=?',(docname,)).fetchone()[0]
    wakeuptimes=wakeuptimes+1
    db.execute('UPDATE docs SET wakeuptimes=? WHERE docname=?',(wakeuptimes,docname))

    #build result
    content=request.args.get('con')
    autocom=get_autocom()
    coms=autocom.doautocom(content)
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
    print(com)
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

@bp.route('/status')
def status():
    db=get_db()
    cursor=db.execute("SELECT * FROM test")
    for row in cursor:
        print("id: %d"%(row[0]))
        print("username: %s"%(row[1]))
    return  redirect(url_for('hello'))