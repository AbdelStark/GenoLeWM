# GenoLeWM Desktop

Reference Tauri 2 shell for RFC-0019. The desktop app is intentionally
skeleton-grade: it proves the local workflow can launch, load the Rust
host, probe the embedded Python runtime, and enforce a narrow network
surface before the scoring UI lands.

## Prerequisites

- Rust stable with Cargo.
- Node.js and pnpm.
- A Python environment where `geno_lewm` is importable if you want the
  runtime probe to report ready.

## Commands

```sh
pnpm install
pnpm build
pnpm tauri dev
```

`pnpm build` validates the TypeScript frontend. From `src-tauri/`, run
`cargo check` to validate the Rust host and PyO3 bridge.

## Network Policy

The Tauri HTTP plugin is configured as default-deny. The only allowed
remote hosts in `src-tauri/capabilities/default.json` are:

- `huggingface.co`
- `*.huggingface.co`
- `ftp.1000genomes.ebi.ac.uk`

The browser CSP mirrors this allowlist. Any host outside that list is
rejected by the Tauri host before it can become an application network
request.
