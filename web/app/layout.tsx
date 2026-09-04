import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADK × AutoCAD — CAD viewer",
  description:
    "Read AutoCAD drawings as structured data, click entities, attach comments.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
