## 🎬 Plugin Demo

See it in action. The video below walks through the BlendAnything interface inside Blender's NLA editor — from loading characters to dialing in a cross-topology blend in just a few clicks.

[MEDIA:demo]

[SPACING:large]

## 🦴 (A) In-Skeleton Blending

Same character, smoother transitions. We compare directly against *Blender's own NLA editor* on same-topology blends — and the difference is stark. Where Blender snaps and pops, BlendAnything flows.

[MEDIA:inSkel1,inSkel2]{**Examples 1–2.** T-Rex (Walk Head Low → Run Roar) and Bat (Kick → Fly). Blender's NLA produces joint-level artifacts and unnatural snapping at the transition point. BlendAnything yields artifact-free, temporally smooth blends in both cases.}

[SPACING:large]

## 🐊 (B) Cross-Skeleton Blending

This is the hard problem — blending across characters that share **no bones whatsoever**. BlendAnything handles it natively inside the NLA editor, with no manual correspondence setup required. The neural backbone finds the shared latent structure automatically.

[MEDIA:xSkel1,xSkel2,xSkel3]{**Examples 1–3.** Hermit Crab → Bear, Raptor → Goat, Crab → Wolf. Each blend transfers not just timing but *intent*: the crab rises and strikes like a bear, the raptor lunges like a goat, the crab charges and leaps like a wolf. Cross-topology semantics preserved throughout.}

[SPACING:large]

## 🎯 (C) Retargeting

Retargeting falls out naturally from the blending framework — set the reference strip's influence to zero, keep only the target, and the model transfers the motion onto an entirely different topology. No extra training, no manual rigging.

[MEDIA:retarget]{🦨 **Skunk (Spray) → Elephant, King Cobra & Tyrannosaurus.** That's stinky! Three radically different characters, one source animation. Style, timing, and the iconic spray posture all transfer cleanly across wildly different body structures.}
