import { build as esbuild } from "esbuild";
import { build as viteBuild } from "vite";
import { rm, readFile, copyFile, mkdir } from "fs/promises";
import { existsSync } from "fs";

// server deps to bundle to reduce openat(2) syscalls
// which helps cold start times
const allowlist = [
  "@google/generative-ai",
  "axios",
  "cors",
  "date-fns",
  "drizzle-orm",
  "drizzle-zod",
  "express",
  "express-rate-limit",
  "express-session",
  "jsonwebtoken",
  "memorystore",
  "multer",
  "nanoid",
  "nodemailer",
  "openai",
  "passport",
  "passport-local",
  "stripe",
  "uuid",
  "ws",
  "xlsx",
  "zod",
  "zod-validation-error",
];

async function buildAll() {
  await rm("dist", { recursive: true, force: true });

  console.log("building client...");
  await viteBuild();

  console.log("building server...");
  const pkg = JSON.parse(await readFile("package.json", "utf-8"));
  const allDeps = [
    ...Object.keys(pkg.dependencies || {}),
    ...Object.keys(pkg.devDependencies || {}),
  ];
  const externals = allDeps.filter((dep) => !allowlist.includes(dep));

  await esbuild({
    entryPoints: ["server/index.ts"],
    platform: "node",
    bundle: true,
    format: "cjs",
    outfile: "dist/index.cjs",
    define: {
      "process.env.NODE_ENV": '"production"',
    },
    minify: true,
    external: externals,
    logLevel: "info",
  });
}

async function copyMarketing() {
  const files = [
    ["marketing/index.html", "dist/public/marketing.html"],
    ["marketing/login.html", "dist/public/login.html"],
  ];
  if (existsSync("marketing/assets/aj.png")) {
    await mkdir("dist/public/assets/marketing", { recursive: true });
    files.push(["marketing/assets/aj.png", "dist/public/assets/marketing/aj.png"]);
  }
  for (const [src, dest] of files) {
    if (existsSync(src)) {
      await copyFile(src, dest);
      console.log(`copied ${src} → ${dest}`);
    }
  }
}

buildAll()
  .then(copyMarketing)
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
