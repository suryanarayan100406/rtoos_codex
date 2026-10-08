export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let detail: unknown;
    try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), response.status);
  }
  return response.json() as Promise<T>;
}
export function fileURL(id: string, path: string, revision?: string | null): string {
  if (!path || path.startsWith("/") || path.includes("\\") || path.includes("\x00") ||
      path.split("/").includes("..")) {
    throw new Error("Invalid bundle-relative artifact path");
  }
  const url = `/api/runs/${encodeURIComponent(id)}/files/${path.split("/").map(encodeURIComponent).join("/")}`;
  return revision ? `${url}?revision=${encodeURIComponent(revision)}` : url;
}

export function downloadURL(id: string, path: string, revision?: string | null): string {
  const url = fileURL(id, path, revision);
  return `${url}${url.includes("?") ? "&" : "?"}download=true`;
}
