import type { FontJob } from "@glyphcraft/shared";
import type { Env } from "../env";

export interface AuthenticatedUser {
  id: string;
  email: string | null;
}

export async function verifySupabaseUser(
  req: Request,
  env: Env
): Promise<AuthenticatedUser | null> {
  const auth = req.headers.get("Authorization");
  if (!auth || !auth.startsWith("Bearer ")) {
    return null;
  }
  const token = auth.slice("Bearer ".length).trim();
  if (!token) return null;

  const resp = await fetch(`${env.SUPABASE_URL.replace(/\/$/, "")}/auth/v1/user`, {
    headers: {
      apikey: env.SUPABASE_ANON_KEY,
      Authorization: `Bearer ${token}`,
    },
  });

  if (!resp.ok) {
    return null;
  }

  const user = (await resp.json()) as { id?: string; email?: string | null };
  if (!user.id) return null;
  return {
    id: user.id,
    email: user.email ?? null,
  };
}

export class SupabaseAdmin {
  private baseUrl: string;
  private serviceKey: string;

  constructor(env: Env) {
    this.baseUrl = `${env.SUPABASE_URL.replace(/\/$/, "")}/rest/v1`;
    this.serviceKey = env.SUPABASE_SERVICE_ROLE_KEY;
  }

  private async request<T>(
    path: string,
    init: RequestInit & { prefer?: string } = {}
  ): Promise<T> {
    const headers: Record<string, string> = {
      apikey: this.serviceKey,
      Authorization: `Bearer ${this.serviceKey}`,
      "Content-Type": "application/json",
    };
    if (init.prefer) {
      headers.Prefer = init.prefer;
    }

    const resp = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        ...headers,
        ...(init.headers as Record<string, string> | undefined),
      },
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Supabase REST error (${resp.status}): ${text}`);
    }

    if (resp.status === 204) {
      return undefined as T;
    }
    return (await resp.json()) as T;
  }

  async ensureProfile(userId: string, email: string | null): Promise<void> {
    await this.request("/profiles?on_conflict=id", {
      method: "POST",
      prefer: "resolution=merge-duplicates,return=minimal",
      body: JSON.stringify({ id: userId, email }),
    });
  }

  async getProfileEmail(userId: string): Promise<string | null> {
    const rows = await this.request<Array<{ email: string | null }>>(
      `/profiles?id=eq.${encodeURIComponent(userId)}&select=email&limit=1`
    );
    return rows[0]?.email ?? null;
  }

  async findOrderByCreemOrderId(
    creemOrderId: string
  ): Promise<{ id: string; user_id: string; status: string } | null> {
    const rows = await this.request<Array<{ id: string; user_id: string; status: string }>>(
      `/orders?creem_order_id=eq.${encodeURIComponent(creemOrderId)}&select=id,user_id,status&limit=1`
    );
    return rows[0] ?? null;
  }

  async insertOrder(input: {
    user_id: string;
    creem_checkout_id: string | null;
    creem_order_id: string;
    amount_cents: number;
    currency: string;
    status: "pending" | "paid" | "refunded";
  }): Promise<{ id: string }> {
    const rows = await this.request<Array<{ id: string }>>("/orders", {
      method: "POST",
      prefer: "return=representation",
      body: JSON.stringify(input),
    });
    return rows[0];
  }

  async findJobByOrderId(orderId: string): Promise<FontJob | null> {
    const rows = await this.request<FontJob[]>(
      `/font_jobs?order_id=eq.${encodeURIComponent(orderId)}&select=*&limit=1`
    );
    return rows[0] ?? null;
  }

  async insertFontJob(input: {
    user_id: string;
    order_id: string;
    draft_id: string;
    family_name: string;
    status: "queued" | "running" | "completed" | "failed";
    progress_message: string;
    ttf_params: {
      threshold: number;
      sidebearing: number;
      curve_mode: "spline" | "polygon";
    };
  }): Promise<FontJob> {
    const rows = await this.request<FontJob[]>("/font_jobs", {
      method: "POST",
      prefer: "return=representation",
      body: JSON.stringify(input),
    });
    return rows[0];
  }

  async updateFontJob(
    jobId: string,
    patch: Partial<{
      status: FontJob["status"];
      progress_message: string;
      family_name: string;
      r2_ttf_key: string | null;
      r2_preview_key: string | null;
      r2_glyphs_zip_key: string | null;
      ttf_params: FontJob["ttf_params"];
      error_message: string | null;
    }>
  ): Promise<FontJob | null> {
    const rows = await this.request<FontJob[]>(
      `/font_jobs?id=eq.${encodeURIComponent(jobId)}`,
      {
        method: "PATCH",
        prefer: "return=representation",
        body: JSON.stringify(patch),
      }
    );
    return rows[0] ?? null;
  }

  async listJobsForUser(userId: string): Promise<FontJob[]> {
    return this.request<FontJob[]>(
      `/font_jobs?user_id=eq.${encodeURIComponent(userId)}&select=*&order=created_at.desc&limit=50`
    );
  }

  async getJobForUser(jobId: string, userId: string): Promise<FontJob | null> {
    const rows = await this.request<FontJob[]>(
      `/font_jobs?id=eq.${encodeURIComponent(jobId)}&user_id=eq.${encodeURIComponent(userId)}&select=*&limit=1`
    );
    return rows[0] ?? null;
  }
}
