// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Get the main window
            let window = app.get_webview_window("main").unwrap();
            
            // You can add custom setup here
            // For example, checking if Python server is running
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Add custom commands here
            check_server_status,
            launch_python_server
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// Command to check if Python server is running
#[tauri::command]
async fn check_server_status() -> Result<bool, String> {
    // Try to connect to localhost:8998
    match reqwest::get("http://localhost:8998").await {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

// Command to launch Python server via WSL
#[tauri::command]
async fn launch_python_server() -> Result<String, String> {
    use std::process::Command;
    
    // Launch the Python server in WSL
    let output = Command::new("wsl")
        .args(&[
            "bash",
            "-c",
            "cd /mnt/c/Users/User/Documents/moonshot/personaplex-desktop/personaplex_server && source venv/bin/activate && export HF_TOKEN='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' && python main.py &"
        ])
        .output()
        .map_err(|e| e.to_string())?;
    
    if output.status.success() {
        Ok("Server started".to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}
