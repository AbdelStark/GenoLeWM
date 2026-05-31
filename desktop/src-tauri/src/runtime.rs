use pyo3::prelude::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeProbe {
    pub python_version: Option<String>,
    pub geno_lewm_available: bool,
    pub runtime_available: bool,
    pub error: Option<String>,
}

pub fn probe_runtime() -> RuntimeProbe {
    Python::attach(|py| {
        let python_version = py
            .import("sys")
            .and_then(|sys| sys.getattr("version"))
            .and_then(|version| version.extract::<String>())
            .ok();

        match py.import("geno_lewm.deploy") {
            Ok(module) => {
                let runtime_available = module.getattr("GenoLeWMRuntime").is_ok();
                RuntimeProbe {
                    python_version,
                    geno_lewm_available: true,
                    runtime_available,
                    error: if runtime_available {
                        None
                    } else {
                        Some("geno_lewm.deploy.GenoLeWMRuntime is not exported".to_string())
                    },
                }
            }
            Err(err) => RuntimeProbe {
                python_version,
                geno_lewm_available: false,
                runtime_available: false,
                error: Some(err.value(py).to_string()),
            },
        }
    })
}
