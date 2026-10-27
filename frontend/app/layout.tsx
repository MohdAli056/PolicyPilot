import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PolicyPilot — AI Policy Assistant",
  description:
    "Ask questions about company policies and receive evidence-grounded answers with document citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
