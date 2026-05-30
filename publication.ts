import type { Publication, Theme } from "@/_internal/types";
import { COMING_SOON } from "@/_internal/types";

const theme: Theme = {
  accentColor:          "#0a4b7c",
  pageBackground:       "#ffffff",
  blockBackground:      "#f7f7f7",
  baseFontSize:         16,
  titleFontSize:        48,
  authorFontSize:       18,
  headingFontSize:      22,
  abstractFontSize:     16,
  contentFontSize:      16,
  mediaTitleFontSize:   16,
  mediaCaptionFontSize: 14,
  contentMaxWidth:      1200,
  bodyFont:             "Lato, sans-serif",
  headingFont:          '"Patua One", serif',
};

const publication: Publication = {
  title: "BlendAnything: A Blender Plugin for Cross-Topology Motion Blending",
  theme,


  authors: [
    ["L. Cazzola",    "https://scholar.google.com/citations?user=fsnsqoYAAAAJ&hl=en", "1"],
    ["G. Martinelli", "https://scholar.google.com/citations?user=WG3OkQ4AAAAJ&hl=en", "1,2"],
    ["N. Conci",      "https://scholar.google.com/citations?user=mR1GK28AAAAJ&hl=en", "1,2"],
  ],

  affiliations: [
    ["1", "University of Trento"],
    ["2", "CNIT"],
  ],

  venue: "SIGGRAPH (Poster) • Special Interest Group on Computer Graphics and Interactive Techniques",
  year:  "2026",

  paper:         COMING_SOON,
  pdf:           undefined,
  code:          "https://github.com/mmlab-cv/BlendAnything",
  supplementary: COMING_SOON,

  siteUrl: "https://github.com/mmlab-cv/",
  siteLabel: "← Check out other MMLab works!",

  teaserIndex: 1,

  abstract: "BlendAnything is a Blender plugin that extends the Non-Linear Animation (NLA) editor with neural cross-topology motion blending. By integrating the learned shared latent representation from Neural Motion Blending directly into Blender's animation workflow, artists can blend motions across characters with entirely different skeletal structures — no manual rigging correspondence required. The plugin supports in-skeleton blending, cross-skeleton blending, and retargeting, all from within the familiar NLA editor interface.",

  media: [
    // 1 — teaser image
    {
      type:    "image",
      src:     "/media/teaser.png",
      id:      "teaser",
      title:   "BlendAnything",
      caption: "BlendAnything extends Blender's NLA editor with neural cross-topology motion blending.",
    },

    // 2 — plugin demo
    {
      type:  "video",
      src:   "/media/plugin_demo.mp4",
      id:    "demo",
      title: "Plugin Demo",
      caption: "A walkthrough of the BlendAnything plugin interface inside Blender's NLA editor.",
      audio: true,
    },

    // (A) In-Skeleton Blending — 3, 4
    {
      type:  "video",
      src:   "/media/inSkel_1.mp4",
      id:    "inSkel1",
      title: "In-Skeleton Example 1",
      caption: "T-Rex (Walk Head Low → Run Roar). Blender produces significant artifacts on leg rotations when the transition begins. Our method yields a smooth, artifact-free blend.",
    },
    {
      type:  "video",
      src:   "/media/inSkel_2.mp4",
      id:    "inSkel2",
      title: "In-Skeleton Example 2",
      caption: "Bat (Kick → Fly). Blender produces snappy, unnatural movement upon transition. Our method produces a natural, fluid blend.",
    },

    // (B) Cross-Skeleton Blending — 5, 6, 7
    {
      type:  "video",
      src:   "/media/xSkel_1.mp4",
      id:    "xSkel1",
      title: "Cross-Skeleton Example 1",
      caption: "Hermit Crab (Attack) → Bear (Rise, Attack). The crab starts with its attacking motion; upon transition, notice how it rises onto its posterior legs and strikes with its claws, faithfully respecting the semantics of the bear's rising attack.",
    },
    {
      type:  "video",
      src:   "/media/xSkel_2.mp4",
      id:    "xSkel2",
      title: "Cross-Skeleton Example 2",
      caption: "Raptor (Roar) → Goat (Head Butt). The raptor begins roaring, then strikes forward just as the goat head-butts, closely matching the goat's tempo and forward-lunge timing.",
    },
    {
      type:  "video",
      src:   "/media/xSkel_3.mp4",
      id:    "xSkel3",
      title: "Cross-Skeleton Example 3",
      caption: "Crab (Attack) → Wolf (Jump Attack). The crab starts attacking, then steps back, charges, and leaps forward. Notice the tight synchronization in the timing of the jump and landing with the wolf's motion.",
    },

    // (C) Retargeting — 8
    {
      type:  "video",
      src:   "/media/transfer.mp4",
      id:    "retarget",
      title: "Retargeting",
      caption: "Skunk (Spray) → Elephant, King Cobra & Tyrannosaurus. The skunk's iconic spray animation is retargeted to three very different characters — our framework handles radically different topologies without breaking a sweat (or a smell).",
    },
  ],
};

export default publication;
