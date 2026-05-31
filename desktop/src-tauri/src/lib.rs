mod runtime;

use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NetworkPolicy {
    allowed_hosts: [&'static str; 2],
    default_deny: bool,
}

#[tauri::command]
fn runtime_probe() -> runtime::RuntimeProbe {
    runtime::probe_runtime()
}

#[tauri::command]
fn network_policy() -> NetworkPolicy {
    NetworkPolicy {
        allowed_hosts: ["huggingface.co", "ftp.1000genomes.ebi.ac.uk"],
        default_deny: true,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![runtime_probe, network_policy])
        .run(tauri::generate_context!())
        .expect("error while running GenoLeWM desktop application");
}
