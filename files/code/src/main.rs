mod storage;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use serde::{Deserialize, Serialize};
use std::{env, io, sync::Arc};
use storage::StorageManager;
use tokio::sync::mpsc;

#[derive(Serialize, Debug, Clone)]
struct GeminiRequest {
    contents: Vec<ContentPayload>,
}

#[derive(Serialize, Debug, Clone)]
struct ContentPayload {
    role: String,
    parts: Vec<PartPayload>,
}

#[derive(Serialize, Debug, Clone)]
struct PartPayload {
    text: String,
}

#[derive(Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    content: Option<Content>,
}

#[derive(Deserialize, Debug)]
struct Content {
    parts: Option<Vec<Part>>,
}

#[derive(Deserialize, Debug)]
struct Part {
    text: Option<String>,
}

enum AppEvent {
    BotResponse(String),
    Error(String),
}

async fn fetch_gemini_response(
    client: reqwest::Client,
    api_key: Arc<String>,
    history: Vec<(String, String)>,
    tx: mpsc::Sender<AppEvent>,
) {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={}",
        api_key
    );

    let contents: Vec<ContentPayload> = history
        .into_iter()
        .filter_map(|(role, text)| {
            let api_role = match role.as_str() {
                "User" => Some("user".to_string()),
                "Bot" => Some("model".to_string()),
                _ => None,
            };

            api_role.map(|r| ContentPayload {
                role: r,
                parts: vec![PartPayload { text }],
            })
        })
        .collect();

    let request_body = GeminiRequest { contents };

    match client.post(&url).json(&request_body).send().await {
        Ok(res) => {
            if res.status().is_success() {
                if let Ok(parsed) = res.json::<GeminiResponse>().await {
                    let text = parsed
                        .candidates
                        .and_then(|c| c.into_iter().next())
                        .and_then(|c| c.content)
                        .and_then(|c| c.parts)
                        .and_then(|p| p.into_iter().next())
                        .and_then(|p| p.text)
                        .unwrap_or_else(|| "[Empty response]".to_string());

                    let _ = tx.send(AppEvent::BotResponse(text)).await;
                } else {
                    let _ = tx.send(AppEvent::Error("Failed to parse API response".to_string())).await;
                }
            } else {
                let err_msg = res.text().await.unwrap_or_else(|_| "Unknown HTTP error".to_string());
                let _ = tx.send(AppEvent::Error(format!("API Error: {}", err_msg))).await;
            }
        }
        Err(e) => {
            let _ = tx.send(AppEvent::Error(format!("Network Error: {}", e))).await;
        }
    }
}

fn handle_command(
    cmd_str: &str,
    current_session: &mut String,
    messages: &mut Vec<(String, String)>,
    storage: &StorageManager,
) -> bool {
    let parts: Vec<&str> = cmd_str.trim().split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    match parts[0] {
        "/session" => {
            if parts.len() < 2 {
                messages.push(("System".to_string(), "Usage: /session [list | new <id> [title] | switch <id>]".to_string()));
                return true;
            }

            match parts[1] {
                "list" => match storage.list_sessions() {
                    Ok(list) => {
                        let mut formatted = String::from("Saved Sessions:\n");
                        for s in list {
                            let mark = if s.id == *current_session { "* " } else { "  " };
                            formatted.push_str(&format!("{}[{}] - {}\n", mark, s.id, s.title));
                        }
                        messages.push(("System".to_string(), formatted.trim_end().to_string()));
                    }
                    Err(e) => messages.push(("System".to_string(), format!("Error listing sessions: {}", e))),
                },
                "new" => {
                    if parts.len() < 3 {
                        messages.push(("System".to_string(), "Usage: /session new <session_id> [optional title]".to_string()));
                    } else {
                        let new_id = parts[2];
                        let title = if parts.len() > 3 { parts[3..].join(" ") } else { new_id.to_string() };

                        match storage.create_session(new_id, &title) {
                            Ok(_) => {
                                *current_session = new_id.to_string();
                                *messages = storage.load_session_history(current_session, 50).unwrap_or_default();
                                messages.push(("System".to_string(), format!("Switched to new session: {}", new_id)));
                            }
                            Err(e) => messages.push(("System".to_string(), format!("Failed to create session: {}", e))),
                        }
                    }
                }
                "switch" => {
                    if parts.len() < 3 {
                        messages.push(("System".to_string(), "Usage: /session switch <session_id>".to_string()));
                    } else {
                        let target_id = parts[2];
                        match storage.session_exists(target_id) {
                            Ok(true) => {
                                *current_session = target_id.to_string();
                                *messages = storage.load_session_history(current_session, 50).unwrap_or_default();
                                messages.push(("System".to_string(), format!("Switched to session: {}", target_id)));
                            }
                            Ok(false) => messages.push(("System".to_string(), format!("Session '{}' not found.", target_id))),
                            Err(e) => messages.push(("System".to_string(), format!("Error checking session: {}", e))),
                        }
                    }
                }
                _ => messages.push(("System".to_string(), "Unknown command. Options: list, new, switch".to_string())),
            }
            true
        }
        "/clear" => {
            messages.clear();
            messages.push(("System".to_string(), format!("Cleared display buffer for session '{}'", current_session)));
            true
        }
        _ => false,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = Arc::new(env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set"));

    let storage_dir = format!("{}/.glox_memory", env::var("HOME").unwrap_or_else(|_| ".".to_string()));
    let storage = StorageManager::new(&storage_dir)?;
    let mut current_session = "main".to_string();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (tx, mut rx) = mpsc::channel::<AppEvent>(32);
    let http_client = reqwest::Client::new();

    let mut input_buffer = String::new();
    let mut messages: Vec<(String, String)> = storage.load_session_history(&current_session, 50).unwrap_or_default();
    let mut is_loading = false;

    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(3)])
                .split(f.size());

            let history_items: Vec<ListItem> = messages
                .iter()
                .map(|(role, msg)| {
                    let (role_color, label) = match role.as_str() {
                        "User" => (Color::Cyan, "User > "),
                        "Bot" => (Color::Green, "Glox > "),
                        _ => (Color::Yellow, "[System] "),
                    };

                    let content = vec![Line::from(vec![
                        Span::styled(label, Style::default().fg(role_color).add_modifier(Modifier::BOLD)),
                        Span::raw(msg),
                    ])];
                    ListItem::new(content)
                })
                .collect();

            let title = format!(" Glox Bot UI | Session: [{}] {} ", current_session, if is_loading { "(Thinking...)" } else { "" });
            let history_block = List::new(history_items).block(Block::default().borders(Borders::ALL).title(title));
            f.render_widget(history_block, chunks[0]);

            let input_widget = Paragraph::new(input_buffer.as_str())
                .style(Style::default().fg(Color::White))
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Prompt (/session, /clear, Esc: exit) "));

            f.render_widget(input_widget, chunks[1]);
            f.set_cursor(chunks[1].x + input_buffer.len() as u16 + 1, chunks[1].y + 1);
        })?;

        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::BotResponse(reply) => {
                    is_loading = false;
                    let _ = storage.record_message(&current_session, "Bot", &reply);
                    messages.push(("Bot".to_string(), reply));
                }
                AppEvent::Error(err) => {
                    is_loading = false;
                    messages.push(("System".to_string(), err));
                }
            }
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Esc => break,
                        KeyCode::Enter => {
                            let raw_input = input_buffer.trim().to_string();
                            if !raw_input.is_empty() && !is_loading {
                                input_buffer.clear();

                                if raw_input.starts_with('/') {
                                    if handle_command(&raw_input, &mut current_session, &mut messages, &storage) {
                                        continue;
                                    }
                                }

                                is_loading = true;
                                let prompt = raw_input.clone();
                                let _ = storage.record_message(&current_session, "User", &prompt);
                                messages.push(("User".to_string(), prompt));

                                let history_snapshot = messages.clone();
                                let tx_clone = tx.clone();
                                let client_clone = http_client.clone();
                                let key_clone = Arc::clone(&api_key);

                                tokio::spawn(async move {
                                    fetch_gemini_response(client_clone, key_clone, history_snapshot, tx_clone).await;
                                });
                            }
                        }
                        KeyCode::Char(c) => input_buffer.push(c),
                        KeyCode::Backspace => { input_buffer.pop(); }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
                                                                                 }
              
