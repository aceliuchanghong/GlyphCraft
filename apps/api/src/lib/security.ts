const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const MAX_REF_PNG_BYTES = 512 * 1024;

export async function timingSafeEqualStr(a: string, b: string): Promise<boolean> {
  const enc = new TextEncoder();
  const [hashA, hashB] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(a)),
    crypto.subtle.digest("SHA-256", enc.encode(b)),
  ]);
  const bytesA = new Uint8Array(hashA);
  const bytesB = new Uint8Array(hashB);
  let diff = 0;
  for (let i = 0; i < bytesA.length; i++) {
    diff |= bytesA[i] ^ bytesB[i];
  }
  return diff === 0;
}

export function getClientIp(req: Request): string {
  const cfIp = req.headers.get("CF-Connecting-IP");
  if (cfIp) return cfIp.trim();
  const xff = req.headers.get("X-Forwarded-For");
  if (xff) return xff.split(",")[0].trim();
  return "127.0.0.1";
}

export function decodeBase64Png(input: string): { bytes: Uint8Array; base64: string } {
  if (typeof input !== "string" || input.length === 0) {
    throw new Error("Empty image payload.");
  }

  const base64 = input.includes(",") ? input.split(",", 2)[1] : input;
  const cleaned = base64.replace(/\s+/g, "");
  if (cleaned.length > MAX_REF_PNG_BYTES * 1.4) {
    throw new Error("Reference image exceeds maximum 512KB size.");
  }

  let binary: string;
  try {
    binary = atob(cleaned);
  } catch {
    throw new Error("Invalid base64 image encoding.");
  }

  if (binary.length < 16 || binary.length > MAX_REF_PNG_BYTES) {
    throw new Error("Reference image must be between 16 bytes and 512KB.");
  }

  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  for (let i = 0; i < PNG_MAGIC.length; i++) {
    if (bytes[i] !== PNG_MAGIC[i]) {
      throw new Error("Reference image must be a valid PNG file.");
    }
  }

  return { bytes, base64: cleaned };
}
