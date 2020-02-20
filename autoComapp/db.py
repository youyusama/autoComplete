import sqlite3
import xlwt

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    db=get_db()
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

@click.command('make-stat')
@with_appcontext
@click.option('--docname',default='')
def make_stat_command(docname):
    db=get_db()
    if docname=='':
        click.echo('docname cannot be null')
    elif db.execute('SELECT * FROM docs WHERE docname=?',(docname,)).fetchone() is None:
        click.echo('docname donot exist')
    else:
        work_book=xlwt.Workbook(encoding='utf-8')
        sheet=work_book.add_sheet(docname)
        doc=db.execute('SELECT edittime,wakeuptimes FROM docs WHERE docname=?',(docname,)).fetchone()
        sheet.write(0,0,'docname:')
        sheet.write(0,1,docname)
        sheet.write(1,0,'doc total edit time:')
        sheet.write(1,1,doc[0])
        sheet.write(2,0,'total autocomplete wakeuptimes in doc:')
        sheet.write(2,1,doc[1])
        sheet.write(4,0,'wakeup info:')
        sheet.write(5, 0, 'wakeuped content')
        sheet.write(5, 1, 'wakeuped times')
        sheetrow=6
        cwakeup=db.execute('SELECT * FROM wakeupcom WHERE indoc=?',(docname,))
        for row in cwakeup:
            sheet.write(sheetrow,0,row[1])
            sheet.write(sheetrow,1,row[2])
            sheetrow=sheetrow+1

        sheetrow=sheetrow+1
        sheet.write(sheetrow, 0, 'chose info:')
        sheetrow=sheetrow+1
        sheet.write(sheetrow, 0, 'chosed content')
        sheet.write(sheetrow, 1, 'edited time')
        sheet.write(sheetrow, 2, 'chosed rank')
        sheet.write(sheetrow, 3, 'edited characters')
        sheetrow = sheetrow + 1
        cchosed=db.execute('SELECT * FROM onechosedcom WHERE indoc=?',(docname,))
        for row in cchosed:
            sheet.write(sheetrow,0,row[1])
            sheet.write(sheetrow, 1, row[2])
            sheet.write(sheetrow, 2, row[3])
            sheet.write(sheetrow, 3, row[4])
            sheetrow=sheetrow+1

        work_book.save('docinfo_%s.xls'%(docname))
        click.echo('statistics output')

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(make_stat_command)