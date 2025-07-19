CREATE SCHEMA IF NOT EXISTS eco;

CREATE TABLE IF NOT EXISTS eco.release_content (
    release_id BIGINT NOT NULL,
    release_name TEXT NOT NULL,
    series_name TEXT NOT NULL,
    bls_series_id VARCHAR NULL,
    fred_series_id VARCHAR NULL,
    source TEXT,
    country_code VARCHAR(3),
    PRIMARY KEY (release_name, series_name)
);

CREATE TABLE IF NOT EXISTS eco.release_schedule (
    release_id BIGINT NOT NULL,
    release_name TEXT,
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    PRIMARY KEY (release_name, date)
);

CREATE TABLE IF NOT EXISTS eco.release_data (
    release_id BIGINT NOT NULL,
    release_name TEXT NOT NULL,
    series_name TEXT NOT NULL,
    release_date DATE NOT NULL,
    reference_date DATE NOT NULL,
    value NUMERIC,
    PRIMARY KEY (release_name, series_name, reference_date),
    FOREIGN KEY (release_name, series_name) REFERENCES eco.release_content(release_name, series_name),
    FOREIGN KEY (release_name, reference_date) REFERENCES eco.release_schedule(release_name, reference_date)
);


