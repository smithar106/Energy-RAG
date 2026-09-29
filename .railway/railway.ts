import { defineRailway, github, postgres, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const Postgres = postgres("Postgres", { region: "us-west2" });
  Postgres.networking = { privateNetworkEndpoint: "postgres" };
  const postgresVolume = volume("postgres-volume", { alerts: { usage: { "100": {}, "80": {}, "95": {} } }, allowOnlineResize: true, region: "us-west2", sizeMB: 50000 });
  const EnergyRAG = service("Energy-RAG", {
    source: github("smithar106/Energy-RAG", { checkSuites: false }),
    builder: "DOCKERFILE",
    dockerfilePath: "Dockerfile",
    start: "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    healthcheck: "/health",
    healthcheckTimeout: 120,
    replicas: { "us-west2": 1 },
    networking: { privateNetworkEndpoint: "energy-rag" },
    env: { ADMIN_API_KEY: preserve(), DATABASE_URL: preserve(), DEEPSEEK_API_KEY: preserve(), DEEPSEEK_MODEL: preserve(), EIA_API_KEY: preserve(), EMBEDDING_MODEL: preserve(), RATE_LIMIT_MAX: preserve(), RATE_LIMIT_WINDOW: preserve() },
  });

  return project("believable-beauty", {
    resources: [EnergyRAG, Postgres, postgresVolume],
  });
});
