CREATE TABLE IF NOT EXISTS "deepmass_user_bak" (
	id TEXT NOT NULL,
	name TEXT NOT NULL,
	contact_info TEXT NOT NULL,
	passwd TEXT NOT NULL
);
CREATE TABLE deepmass_login (
	contact_info TEXT NOT NULL,
	login_time REAL
);
CREATE TABLE deepmass_code (
	contact_info TEXT NOT NULL,
	verify_code TEXT
, verify_time REAL);
CREATE TABLE deepmass_work (
	id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	email TEXT NOT NULL,
	spectrum_count INTEGER,
	submit_time REAL,
	end_time REAL,
	work_duration REAL
);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE IF NOT EXISTS "deepmass_user" (
    id TEXT PRIMARY KEY NOT NULL,      
    name TEXT NOT NULL,
    contact_info TEXT NOT NULL UNIQUE,
    passwd TEXT NOT NULL                 
);
