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

type FileSlot = {
  kind: "vcf" | "fasta";
  dropId: string;
  inputId: string;
  labelId: string;
};

const FILE_SLOTS: FileSlot[] = [
  { kind: "vcf", dropId: "vcf-drop", inputId: "vcf-input", labelId: "vcf-filename" },
  { kind: "fasta", dropId: "fasta-drop", inputId: "fasta-input", labelId: "fasta-filename" },
];

const selectedInputs = new Set<FileSlot["kind"]>();

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

const renderQueueStatus = () => {
  if (selectedInputs.size === 0) {
    text("queue-status", "Waiting for local inputs.");
  } else if (selectedInputs.size === FILE_SLOTS.length) {
    text("queue-status", "VCF and FASTA selected. Scoring action is pending runtime wiring.");
  } else {
    text("queue-status", `${selectedInputs.size} of ${FILE_SLOTS.length} local inputs selected.`);
  }
};

const setSelectedFile = (slot: FileSlot, file: File) => {
  selectedInputs.add(slot.kind);
  text(slot.labelId, file.name);
  renderQueueStatus();
};

const markDropTarget = (event: DragEvent, slot: FileSlot) => {
  event.preventDefault();
  const first = event.dataTransfer?.files.item(0);
  if (first) {
    setSelectedFile(slot, first);
  }
};

const bindFileSlot = (slot: FileSlot) => {
  const drop = document.querySelector<HTMLElement>(`#${slot.dropId}`);
  const input = document.querySelector<HTMLInputElement>(`#${slot.inputId}`);
  const picker = document.querySelector<HTMLButtonElement>(`[data-picker="${slot.inputId}"]`);

  picker?.addEventListener("click", () => input?.click());
  input?.addEventListener("change", () => {
    const first = input.files?.item(0);
    if (first) {
      setSelectedFile(slot, first);
    }
  });
  drop?.addEventListener("dragover", (event) => event.preventDefault());
  drop?.addEventListener("drop", (event) => markDropTarget(event, slot));
};

window.addEventListener("DOMContentLoaded", async () => {
  FILE_SLOTS.forEach(bindFileSlot);

  try {
    renderNetworkPolicy(await invoke<NetworkPolicy>("network_policy"));
    renderRuntime(await invoke<RuntimeProbe>("runtime_probe"));
  } catch (error) {
    text("runtime-pill", "Runtime error");
    text("runtime-error", String(error));
  }
});
