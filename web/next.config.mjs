/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produces .next/standalone — a self-contained server with only the
  // packages actually imported, so the runtime image does not ship
  // node_modules.
  output: "standalone",
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
