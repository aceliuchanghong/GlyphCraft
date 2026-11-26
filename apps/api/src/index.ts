import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Env } from "./env";
import { checkoutRoute } from "./routes/checkout";
import { draftsRoute } from "./routes/drafts";
import { internalRoute } from "./routes/internal";
import { jobsRoute } from "./routes/jobs";
import { previewRoute } from "./routes/preview";
import { webhooksRoute } from "./routes/webhooks";

const app = new Hono<{ Bindings: Env }>();

app.use("/api/*", async (c, next) => {
  const configuredOrigin = c.env.FRONTEND_URL?.replace(/\/$/, "");
  const middleware = cors({
    origin: (origin) => {
      if (!origin) return configuredOrigin || "*";
      if (origin === configuredOrigin) return origin;
      if (
        c.env.ENVIRONMENT === "development" &&
        /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
      ) {
        return origin;
      }
      return configuredOrigin || "";
    },
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    exposeHeaders: ["X-Remaining-Quota", "X-Quota-Limit"],
    maxAge: 86400,
  });
  return middleware(c, next);
});

app.get("/health", (c) => c.json({ ok: true, service: "glyphcraft-api" }));

app.route("/api/preview", previewRoute);
app.route("/api/drafts", draftsRoute);
app.route("/api/checkout", checkoutRoute);
app.route("/api/webhooks", webhooksRoute);
app.route("/api/internal", internalRoute);
app.route("/api/jobs", jobsRoute);

app.notFound((c) => c.json({ error: "Route not found." }, 404));

app.onError((err, c) => {
  console.error("[api-error]", err);
  return c.json(
    {
      error:
        c.env.ENVIRONMENT === "development"
          ? err.message
          : "Internal server error.",
    },
    500
  );
});

export default app;
