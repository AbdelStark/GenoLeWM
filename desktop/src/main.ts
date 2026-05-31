import { invoke } from "@tauri-apps/api/core";

type RuntimeProbe = {
  pythonVersion: string | null;
  genoLewmAvailable: boolean;
  runtimeAvailable: boolean;
  error: string | null;
};

type NetworkPolicy = {
  allowedHosts: string[];
  defaultDeny: boolean;
};

const text = (id: string, value: string) => {
  const node = document.querySelector<HTMLElement>(`#${id}`);
  if (node) {
    node.textContent = value;
  }
};

const renderRuntime = (probe: RuntimeProbe) => {
  text("runtime-pill", probe.runtimeAvailable ? "Runtime ready" : "Runtime unavailable");
  text("runtime-python", probe.pythonVersion?.split(" ")[0] ?? "unavailable");
  text("runtime-package", probe.genoLewmAvailable ? "available" : "missing");
  text("runtime-deploy", probe.runtimeAvailable ? "available" : "missing");
  text("runtime-error", probe.error ?? "");
};

const renderNetworkPolicy = (policy: NetworkPolicy) => {
  const list = document.querySelector("#network-hosts");
  if (list) {
    list.replaceChildren(
      ...policy.allowedHosts.map((host) => {
        const item = document.createElement("li");
        item.textContent = host;
        return item;
      }),
    );
  }
  text("network-mode", policy.defaultDeny ? "default deny" : "default allow");
};

const markDropTarget = (event: DragEvent, id: string) => {
  event.preventDefault();
  const first = event.dataTransfer?.files.item(0);
  if (!first) {
    return;
  }
  const target = document.querySelector(`#${id} strong`);
  if (target) {
    target.textContent = first.name;
  }
};

window.addEventListener("DOMContentLoaded", async () => {
  for (const id of ["vcf-drop", "fasta-drop"]) {
    const drop = document.querySelector<HTMLElement>(`#${id}`);
    drop?.addEventListener("dragover", (event) => event.preventDefault());
    drop?.addEventListener("drop", (event) => markDropTarget(event, id));
  }

  try {
    renderNetworkPolicy(await invoke<NetworkPolicy>("network_policy"));
    renderRuntime(await invoke<RuntimeProbe>("runtime_probe"));
  } catch (error) {
    text("runtime-pill", "Runtime error");
    text("runtime-error", String(error));
  }
});
