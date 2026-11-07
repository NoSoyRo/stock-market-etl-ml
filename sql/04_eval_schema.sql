-- Producto de datos: Evaluacion / Riesgo
-- Motor: SQL Server
-- Version inicial: 1.0

IF NOT EXISTS (
    SELECT 1
    FROM sys.schemas
    WHERE name = 'eval'
)
BEGIN
    EXEC('CREATE SCHEMA eval');
END;
GO

-- Registro de ejecuciones de backtesting
IF OBJECT_ID('eval.backtest_runs', 'U') IS NULL
BEGIN
    CREATE TABLE eval.backtest_runs (
        run_id INT IDENTITY(1,1) PRIMARY KEY,
        universe NVARCHAR(MAX) NOT NULL,
        generated_at DATETIME2 NOT NULL,
        horizon INT NULL,
        top_k INT NULL,
        score_mode NVARCHAR(50) NULL,
        weight_mode NVARCHAR(50) NULL,
        train_start DATE NULL,
        train_end DATE NULL,
        test_start DATE NULL,
        test_end DATE NULL,
        n_train INT NULL,
        n_test INT NULL,
        n_periods INT NULL,
        created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT ck_eval_top_k
            CHECK (top_k IS NULL OR top_k > 0),

        CONSTRAINT ck_eval_n_periods
            CHECK (n_periods IS NULL OR n_periods >= 0)
    );
END;
GO

-- Metricas de rendimiento y riesgo
IF OBJECT_ID('eval.risk_metrics', 'U') IS NULL
BEGIN
    CREATE TABLE eval.risk_metrics (
        metric_id INT IDENTITY(1,1) PRIMARY KEY,
        run_id INT NOT NULL UNIQUE,
        sharpe_gross FLOAT NULL,
        sharpe_net FLOAT NULL,
        max_drawdown FLOAT NULL,
        calmar_ratio FLOAT NULL,
        win_rate_net FLOAT NULL,
        total_return_gross FLOAT NULL,
        total_return_net FLOAT NULL,
        auc FLOAT NULL,
        commission_pct FLOAT NULL,
        slippage_pct FLOAT NULL,
        round_trip_cost FLOAT NULL,
        deflated_sharpe FLOAT NULL,

        CONSTRAINT fk_eval_risk_run
            FOREIGN KEY (run_id)
            REFERENCES eval.backtest_runs(run_id),

        CONSTRAINT ck_eval_auc
            CHECK (auc IS NULL OR auc BETWEEN 0 AND 1),

        CONSTRAINT ck_eval_win_rate
            CHECK (win_rate_net IS NULL OR win_rate_net BETWEEN 0 AND 1),

        CONSTRAINT ck_eval_dsr
            CHECK (deflated_sharpe IS NULL OR deflated_sharpe BETWEEN 0 AND 1)
    );
END;
GO

-- Bitacora de evaluaciones y experimentos
IF OBJECT_ID('eval.experiment_log', 'U') IS NULL
BEGIN
    CREATE TABLE eval.experiment_log (
        log_id INT IDENTITY(1,1) PRIMARY KEY,
        run_id INT NULL,
        experiment_name NVARCHAR(150) NOT NULL,
        configuration_json NVARCHAR(MAX) NULL,
        results_json NVARCHAR(MAX) NULL,
        status NVARCHAR(30) NOT NULL DEFAULT 'COMPLETED',
        created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT fk_eval_log_run
            FOREIGN KEY (run_id)
            REFERENCES eval.backtest_runs(run_id),

        CONSTRAINT ck_eval_configuration_json
            CHECK (
                configuration_json IS NULL
                OR ISJSON(configuration_json) = 1
            ),

        CONSTRAINT ck_eval_results_json
            CHECK (
                results_json IS NULL
                OR ISJSON(results_json) = 1
            )
    );
END;
GO
