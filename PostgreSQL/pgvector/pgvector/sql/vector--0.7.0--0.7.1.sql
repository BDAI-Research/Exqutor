-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.7.1'" to load this file. \quit

CREATE TABLE IF NOT EXISTS pgvector_qerror (
    table_name TEXT PRIMARY KEY,
    sample_size FLOAT,
    recent_qerrors FLOAT[],
    qerror_count INT,
    v_grad FLOAT,
    learning_rate FLOAT
);