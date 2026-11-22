//! Open the local database and apply schema changes under a write lock.
use rusqlite::{Connection, TransactionBehavior};
use std::{error::Error, fs, path::Path, time::Duration};

const SCHEMA_VERSION: i32 = 1;

pub fn initialize(directory: &Path) -> Result<i32, Box<dyn Error + Send + Sync>> {
    fs::create_dir_all(directory)?;
    let mut connection = Connection::open(directory.join("studio.sqlite3"))?;
    connection.busy_timeout(Duration::from_secs(2))?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let version: i32 = transaction.pragma_query_value(None, "user_version", |row| row.get(0))?;
    match version {
        0 => transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?,
        SCHEMA_VERSION => {}
        other => return Err(format!("Unsupported database schema {other}").into()),
    }
    transaction.commit()?;
    Ok(SCHEMA_VERSION)
}
