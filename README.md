# MarkDownPAPER 📄

From Markdown to published paper website — in minutes.

---

## 🚀 Quickstart

### 1 — Clone this repository

```bash
git clone --recursive https://github.com/LuCazzola/MarkDownPAPER my-paper
cd my-paper
```

> `--recursive` is required to fetch the `md-paper` engine submodule.

### 2 — Install dependencies

```bash
npm install
```

### 3 — Edit your paper ✏️

The only files you need to touch:

```
my-paper/
├── public/
│   └── media/         ← 🖼️  drop your images and videos here
├── publication.ts     ← 📋  title, authors, links, media list, theme
└── content.md         ← 📝  paper body in Markdown
```

Preview live as you edit:

```bash
npm run dev
```

Open `http://localhost:5173`.

### 4 — Build 🔨

```bash
npm run build
```

Output goes to `docs/`.

### 5 — Deploy to GitHub Pages 🌐

Rename the remote to point to your own paper repo:

```bash
git remote set-url origin https://github.com/your-org/my-paper
```

Commit everything and push:

```bash
git add -A
git commit -m "Update paper"
git push origin main
```

On GitHub: **Settings → Pages → Source → Deploy from branch**, select your branch and `/docs`.

> ⚠️ **Important:** open `package.json` and set `"name"` to match your GitHub repository name exactly — this is used as the base path for all assets. If the name is wrong the deployed site will show a blank page.
> ```json
> { "name": "your-repo-name", ... }
> ```

---

## 🔄 Updating the engine

When [md-paper](https://github.com/LuCazzola/md-paper) releases an update:

```bash
git submodule update --remote md-paper
git add md-paper
git commit -m "Update md-paper engine"
git push origin main
```

Your `publication.ts`, `content.md`, and `public/media/` are never touched.

---

## 📖 Reference

### `publication.ts` — metadata, links, media list, theme

```ts
import type { Publication, Theme } from "./md-paper/types";
import { COMING_SOON } from "./md-paper/types";

const theme: Theme = {
  accentColor:          "#0a4b7c",   // headings and accent color
  pageBackground:       "#ffffff",
  blockBackground:      "#f7f7f7",   // abstract block background
  baseFontSize:         16,          // scales the whole page
  titleFontSize:        48,
  authorFontSize:       18,
  headingFontSize:      22,
  abstractFontSize:     16,
  contentFontSize:      16,
  mediaTitleFontSize:   18,
  mediaCaptionFontSize: 13,
  contentMaxWidth:      1200,
  bodyFont:             "Lato, sans-serif",
  headingFont:          '"Patua One", serif',
};

const publication: Publication = {
  title: "Your Paper Title",
  theme,

  authors: [
    ["A. Author", "https://scholar.google.com/...", "1"],
    ["B. Coauthor", undefined, "1,2"],   // no URL → plain text
  ],
  affiliations: [
    ["1", "University X"],
    ["2", "Institute Y"],
  ],

  venue: "CVPR 2025",   // or undefined to hide
  year:  "2025",
  abstract: "Your abstract goes here.",

  // "https://..." → active link   COMING_SOON → greyed out   undefined → hidden
  paper:         "https://arxiv.org/abs/XXXX.XXXXX",
  pdf:           undefined,
  code:          COMING_SOON,
  supplementary: undefined,

  siteUrl:     "https://your-portfolio.github.io/",
  teaserIndex: 1,   // which media item to show as the teaser (1-based)

  // Drop files in public/media/ and reference them with /media/filename
  media: [
    {
      type:    "image",
      src:     "/media/figure1.png",
      id:      "overview",   // optional alias — use [MEDIA:overview] in content.md
      title:   "Figure title",
      caption: "Figure caption.",
    },
    {
      type:  "video",
      src:   "/media/demo.mp4",
      title: "Demo",
    },
  ],
};

export default publication;
```

---

### `content.md` — paper body

Standard Markdown with special tokens for embedding media.

#### 🖼️ Embedding media

| Token | Result |
|---|---|
| `[MEDIA:1]` | Single item, full width |
| `[MEDIA:overview]` | Single item by alias |
| `[MEDIA:1:0.6]` | Single item at 60% width |
| `[MEDIA:1-4]` | Carousel — items 1 through 4 |
| `[MEDIA:1,3,5]` | Carousel — non-contiguous picks |

Append `{...}` for a Markdown caption rendered above the media:

```
[MEDIA:1]{**Figure 1.** Supports **bold**, *italic*, `code`, and math ($\alpha$).}
```

#### 🗂️ Multi-column layout

```
[MEDIA-MULTICOL:1.1]
[MEDIA:1]{Left caption.}
[MEDIA:2]{Right caption.}
[/MEDIA-MULTICOL]
```

The scale factor (`1.1`) lets the block extend beyond the content width. Columns collapse to a single column on mobile.

#### 📐 Spacing

```
[SPACING:small]    →  16 px
[SPACING:medium]   →  32 px
[SPACING:large]    →  48 px
[SPACING:xlarge]   →  64 px
```

#### 🔢 Math

Full KaTeX — inline `$...$` and display `$$...$$`.

---

## License

Open source — free to use with attribution.  
If md-paper is useful for your work, a link back is appreciated. ⭐
