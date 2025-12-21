-- Ledger schema 스켈레톤
CREATE TABLE IF NOT EXISTS experiments (
    exp_id TEXT PRIMARY KEY,
    policy_spec JSONB NOT NULL,
    data_hash TEXT NOT NULL,
    feature_hash TEXT NOT NULL,
    label_hash TEXT NOT NULL,
    model_artifact_ref TEXT,
    cpcv_metrics JSONB,
    pbo REAL,
    risk_report JSONB,
    reason_codes JSONB,
    fix_suggestion JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
