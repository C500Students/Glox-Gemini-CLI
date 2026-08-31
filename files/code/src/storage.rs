use fjall::{Config, Keyspace, PartitionHandle};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const SESSIONS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("sessions_meta");

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct StorageManager {
    redb: Arc<Database>,
    _keyspace: Keyspace,
    chat_partition: PartitionHandle,
}

impl StorageManager {
    pub fn new(base_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let db_dir = std::path::Path::new(base_path);
        std::fs::create_dir_all(db_dir)?;

        let redb = Database::create(db_dir.join("meta.redb"))?;
        let write_txn = redb.begin_write()?;
        {
            let _ = write_txn.open_table(SESSIONS_TABLE)?;
        }
        write_txn.commit()?;

        let keyspace = Config::new(db_dir.join("history.fjall")).open()?;
        let chat_partition = keyspace.open_partition("messages", Default::default())?;

        let manager = Self {
            redb: Arc::new(redb),
            _keyspace: keyspace,
            chat_partition,
        };

        if manager.list_sessions()?.is_empty() {
            manager.create_session("main", "Default Session")?;
        }

        Ok(manager)
    }

    pub fn create_session(&self, id: &str, title: &str) -> Result<SessionMeta, Box<dyn std::error::Error>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let meta = SessionMeta {
            id: id.to_string(),
            title: title.to_string(),
            created_at: now,
        };

        let encoded = bincode::serialize(&meta)?;
        let write_txn = self.redb.begin_write()?;
        {
            let mut table = write_txn.open_table(SESSIONS_TABLE)?;
            table.insert(id, encoded.as_slice())?;
        }
        write_txn.commit()?;

        Ok(meta)
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionMeta>, Box<dyn std::error::Error>> {
        let read_txn = self.redb.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;
        let mut sessions = Vec::new();

        for entry in table.iter()? {
            let (_k, v) = entry?;
            if let Ok(meta) = bincode::deserialize::<SessionMeta>(v.value()) {
                sessions.push(meta);
            }
        }

        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    pub fn session_exists(&self, id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let read_txn = self.redb.begin_read()?;
        let table = read_txn.open_table(SESSIONS_TABLE)?;
        Ok(table.get(id)?.is_some())
    }

    pub fn record_message(&self, session_id: &str, role: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;

        let key = format!("{}:{:020}", session_id, now);
        let msg = StoredMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: now,
        };

        let encoded = bincode::serialize(&msg)?;
        self.chat_partition.insert(key.as_bytes(), encoded)?;
        self.chat_partition.persist()?;
        Ok(())
    }

    pub fn load_session_history(&self, session_id: &str, limit: usize) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
        let prefix = format!("{}:", session_id);
        let mut turns = Vec::new();

        for item in self.chat_partition.prefix(prefix.as_bytes()) {
            let (_k, v) = item?;
            if let Ok(msg) = bincode::deserialize::<StoredMessage>(&v) {
                turns.push((msg.role, msg.content));
            }
        }

        let start = if turns.len() > limit { turns.len() - limit } else { 0 };
        Ok(turns[start..].to_vec())
    }
              }
