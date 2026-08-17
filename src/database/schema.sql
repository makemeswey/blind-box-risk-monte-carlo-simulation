
-- Create figures table
CREATE TABLE IF NOT EXISTS figures (
    figure_id             VARCHAR NOT NULL,
    collection_id         VARCHAR NOT NULL,
    collection_name       VARCHAR NOT NULL,
    figure_name           VARCHAR NOT NULL,
    rarity                VARCHAR NOT NULL,   
    probability_raw       DOUBLE  NOT NULL,   
    probability           DOUBLE  NOT NULL,   
    normalisation_factor  DOUBLE  NOT NULL,   
    box_price             DOUBLE  NOT NULL,
    currency              VARCHAR NOT NULL,
    source                VARCHAR,
    PRIMARY KEY (collection_id, figure_id)
);

-- Create simulation runs table
CREATE SEQUENCE IF NOT EXISTS seq_run_id START 1;

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id           INTEGER PRIMARY KEY DEFAULT nextval('seq_run_id'),
    collection_id    VARCHAR   NOT NULL,
    num_simulations  INTEGER   NOT NULL,
    budget           DOUBLE,
    box_price        DOUBLE    NOT NULL,
    seed             BIGINT,
    params           VARCHAR,              
    created_at       TIMESTAMP NOT NULL DEFAULT now()
);

-- Create simulation results table 
CREATE TABLE IF NOT EXISTS simulation_results (
    run_id          INTEGER NOT NULL,
    simulation_id   INTEGER NOT NULL,
    boxes_required  INTEGER NOT NULL,
    duplicates      INTEGER NOT NULL,
    total_cost      DOUBLE  NOT NULL,
    PRIMARY KEY (run_id, simulation_id)
);
