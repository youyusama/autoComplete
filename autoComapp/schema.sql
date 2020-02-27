DROP TABLE IF EXISTS docs;
DROP TABLE IF EXISTS wakeupcom;
DROP TABLE IF EXISTS chosedcom;
DROP TABLE IF EXISTS onechosedcom;

CREATE TABLE docs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    docname TEXT NOT NULL,
    edittime INTEGER NOT NULL,
    wakeuptimes INTEGER NOT NULL
);

CREATE TABLE wakeupcom(
    indoc TEXT NOT NULL,
    com TEXT NOT NULL,
    times INTEGER NOT NULL
);

CREATE TABLE chosedcom(
    indoc TEXT NOT NULL,
    com TEXT NOT NULL,
    edittime INTEGER NOT NULL,
    ranking INTEGER NOT NULL,
    edited INTEGER NOT NULL,
    times INTEGER NOT NULL
);

CREATE TABLE onechosedcom(
    indoc TEXT NOT NULL,
    com TEXT NOT NULL,
    edittime INTEGER NOT NULL,
    ranking INTEGER NOT NULL,
    edited INTEGER NOT NULL
);