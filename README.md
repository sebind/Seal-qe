<div align="center">
  <h1>🔬 Seal — Quantum ESPRESSO Input Generator</h1>
  <p>A high-performance visual interface for generating Quantum ESPRESSO input files from crystal structure data.</p>

  <p>
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#deploy-to-vercel">Deploy</a> •
    <a href="#tech-stack">Tech Stack</a>
  </p>
</div>

---

## Features

- **Multi-format parsing** — Import `.cif`, `.xyz`, `.vasp` (POSCAR/CONTCAR), `.xsf`, and QE `.in` files
- **Fast local parsing** — Instant client-side parsing for common formats; Gemini AI fallback for complex CIF files
- **3D structure viewer** — Interactive React Three Fiber visualization with atom labels, bonds, and unit cell
- **Full pw.x configuration** — Control, System, Electrons, Ions, Cell namelists with DFT+U / Hubbard support
- **pp.x & ph.x generators** — Post-processing and phonon input file generation
- **Preset structures** — Silicon, Graphene, TiO₂ (Rutile), SrTiO₃ (Perovskite)
- **One-click export** — Copy or download generated input files

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- A [Gemini API key](https://aistudio.google.com/apikey) (optional — only needed for AI-powered CIF parsing)

### Install & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/seal-quantum-espresso-input-generator.git
cd seal-quantum-espresso-input-generator

# Install dependencies
npm install

# (Optional) Set your Gemini API key for AI-powered parsing
cp .env.local.example .env.local
# Edit .env.local and add your key

# Start the dev server
npm run dev
```

The app will be available at **http://localhost:3000**.

## Deploy to Vercel

1. Push this repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new) and import the repo
3. Add environment variable `VITE_GEMINI_API_KEY` in Vercel dashboard → Settings → Environment Variables
4. Deploy — Vercel auto-detects the Vite framework

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/seal-quantum-espresso-input-generator)

> **Note:** The Gemini API key is embedded in the client-side bundle. This is fine for personal or demo use. For production, consider adding a serverless API route to proxy Gemini calls.

## Tech Stack

- **React 19** + **TypeScript** — UI framework
- **Vite** — Build tool & dev server
- **Tailwind CSS v4** — Styling
- **React Three Fiber** — 3D crystal structure visualization
- **Google Gemini API** — AI-powered crystal structure parsing (optional)
- **Lucide React** — Icons

## Project Structure

```
├── public/               # Static assets
├── src/
│   ├── App.tsx           # Main application
│   ├── main.tsx          # React entry point
│   ├── index.css         # Tailwind CSS entry
│   ├── types.ts          # TypeScript types
│   ├── constants.ts      # QE defaults, element data
│   ├── components/
│   │   ├── SettingsPanel.tsx    # QE parameter controls
│   │   └── StructureViewer.tsx  # 3D visualization
│   ├── services/
│   │   └── geminiService.ts     # Gemini AI integration
│   └── utils/
│       ├── latticeUtils.ts      # Reciprocal lattice math
│       ├── localParsers.ts      # CIF/XYZ/VASP/XSF parsers
│       ├── qeGenerator.ts       # QE input file generation
│       └── symmetryUtils.ts     # High-symmetry points
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── vercel.json
```

## License

MIT
