-- Laboratorio de mercado — Postgres 16.
-- Ejecutar ya conectado a la base `mercado`
-- (Cloud SQL domesticos-pg, proyecto domesticospro).
-- Crea únicamente el esquema lab. No toca dbo ni objetos de otras bases.
-- Cadencia: los días hábiles llenan price+metric+news; el fin de semana solo news.

DO $$
BEGIN
    IF current_database() IS DISTINCT FROM 'mercado' THEN
        RAISE EXCEPTION
            '03_lab_schema.sql solo aplica en la base mercado (conectado a: %)',
            current_database();
    END IF;
END
$$;

CREATE SCHEMA IF NOT EXISTS lab;

COMMENT ON SCHEMA lab IS
    'Laboratorio de mercado. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.price_daily (
    ticker      text        NOT NULL,
    trade_date  date        NOT NULL,
    open        numeric,
    high        numeric,
    low         numeric,
    close       numeric,
    volume      numeric,
    moneda      text,
    source      text,
    loaded_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);

COMMENT ON TABLE lab.price_daily IS
    'Precios OHLCV. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.metric_daily (
    ticker      text             NOT NULL,
    trade_date  date             NOT NULL,
    feature     text             NOT NULL,
    value       double precision,
    woe         double precision,
    iv          double precision,
    PRIMARY KEY (ticker, trade_date, feature)
);

COMMENT ON TABLE lab.metric_daily IS
    'Features diarias (value, woe, iv). Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.news_daily (
    id          serial PRIMARY KEY,
    ticker      text             NOT NULL,
    trade_date  date             NOT NULL,
    title       text,
    source      text,
    sentiment   double precision,
    url         text
);

COMMENT ON TABLE lab.news_daily IS
    'Noticias, varias por ticker y día. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.news_feature_daily (
    ticker              text             NOT NULL,
    trade_date          date             NOT NULL,
    universe            text,
    news_sent_score     double precision,
    news_count_7d       double precision,
    news_sent_delta     double precision,
    news_sent_vol       double precision,
    news_relevance_avg  double precision,
    news_esg_score      double precision,
    n_articles          integer,
    sources             text,
    loaded_at           timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);

COMMENT ON TABLE lab.news_feature_daily IS
    'Features agregadas de noticias por ticker y fecha (Golden). Grano ticker × trade_date.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.decision (
    id           serial PRIMARY KEY,
    decided_at   timestamptz NOT NULL DEFAULT now(),
    session_date date,
    model        text,
    cash_usd     numeric,
    motivo       text,
    raw          jsonb
);

COMMENT ON TABLE lab.decision IS
    'Decisión de la sesión. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.entry_plan (
    id          serial PRIMARY KEY,
    decision_id integer NOT NULL REFERENCES lab.decision (id),
    ticker      text    NOT NULL,
    usd         numeric,
    horizon     text,
    stop_usd    numeric,
    p_sube      numeric,
    enter_on    date
);

COMMENT ON TABLE lab.entry_plan IS
    'Plan de entrada de una decisión. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.outcome (
    id              serial PRIMARY KEY,
    entry_plan_id   integer     NOT NULL REFERENCES lab.entry_plan (id),
    scored_at       timestamptz NOT NULL DEFAULT now(),
    realized_return numeric,
    hit_stop        boolean,
    note            text
);

COMMENT ON TABLE lab.outcome IS
    'Resultado realizado de un plan. Días hábiles llenan price+metric+news; fin de semana solo news.';

-- Días hábiles llenan price+metric+news; fin de semana solo news.
CREATE TABLE IF NOT EXISTS lab.run_log (
    id          serial PRIMARY KEY,
    job         text        NOT NULL,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status      text,
    detail      text
);

COMMENT ON TABLE lab.run_log IS
    'Bitácora de jobs. Días hábiles llenan price+metric+news; fin de semana solo news.';

CREATE INDEX IF NOT EXISTS news_daily_ticker_date_idx
    ON lab.news_daily (ticker, trade_date);

CREATE INDEX IF NOT EXISTS news_feature_daily_date_idx
    ON lab.news_feature_daily (trade_date);

CREATE INDEX IF NOT EXISTS decision_session_date_idx
    ON lab.decision (session_date);

CREATE INDEX IF NOT EXISTS entry_plan_decision_id_idx
    ON lab.entry_plan (decision_id);

CREATE INDEX IF NOT EXISTS outcome_entry_plan_id_idx
    ON lab.outcome (entry_plan_id);

CREATE INDEX IF NOT EXISTS run_log_started_at_idx
    ON lab.run_log (started_at);
