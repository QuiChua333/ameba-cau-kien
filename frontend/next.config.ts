import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["192.168.*.*"],
  experimental: {
    serverActions: {
      allowedOrigins: ["192.168.*.*"],
    },
  },
};

export default nextConfig;
