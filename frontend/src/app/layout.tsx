import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aegis RAG — Autonomous Retrieval Engine",
  description: "Production-grade agentic knowledge retrieval and cognitive reasoning platform.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
