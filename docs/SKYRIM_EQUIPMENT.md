# Skyrim humanoid NPC equipment

Equipment support is partial. The runtime now connects inventory ownership,
visible NPC geometry and attachment timing, but full retail behavior graph
execution and FNIS-generated output acceptance remain outstanding.

## Implemented

- Persistent biped, right-hand, left-hand and ammunition slot masks. Equipping
  a weapon preserves clothing; shields conflict with two-handed weapons.
- Script equipment/removal policies, equip/unequip and equipment queries.
  Draw/sheath and named animation requests use the shared session interface.
- Skyrim NPC outfit materialization occurs once. Later geometry reads the
  equipped inventory, including explicit empty equipment. Outfit changes rebuild
  the worn set; removal/transfer of equipped items removes their visible geometry.
- Profile-resolved weapon models attach to actual skeleton bones. Retail
  `WEAPON`, `SHIELD`, `QUIVER` and weapon-family sheath nodes are case-sensitive.
  Missing custom attachment nodes produce diagnostics instead of root binding.
  Rigid `Scb` shapes remain at the sheath attachment while the blade moves to
  the hand; compound scabbards still require separate validation.
- Equipped melee contact waits for drawing to finish. Combat-owned draws
  sheath after combat ends; explicit script draws remain under script control.
- Draw/sheath attachment changes consume timeline `weaponDraw`/`weaponSheathe`
  events. Request events cannot directly move equipment. Duplicate timeline
  markers are ignored after consumption; interruptions cancel pending handoffs.
- Geometry changes preserve graph instances. A per-bone bind correction reconciles
  the retained animation palette with a rebuilt mesh's inverse bind matrices.
- ARMA weight-slider flags and NPC NAM7 weight select compatible `_0`/`_1`
  interpolation. Topology/bind validation precedes any mutation; influences are
  combined by bone identity, reduced deterministically and normalized.
- Headgear can hide HDPT hair and its extra parts, plus isolated hair/ear NIF
  partitions. The head slot alone does not remove the face.
- Naked-body selection uses anatomical slots rather than demanding coverage of
  every auxiliary jewelry/limb slot. This removes the duplicate naked torso
  beneath Ralof's Stormcloak cuirass while retaining exposed feet/hands.
- Save version 12 retains slot ownership, script policies and pending attachment
  state. Older saves default to one-time NPC materialization. Incompatible saved
  animation fingerprints are invalidated once with a diagnostic; equipment
  ownership survives and stale melee/attachment effects are cancelled.

No cooked-scene or streamed-chunk layout changes are required.

## Reference and verification

Record layouts follow [xEdit's TES5 definitions](https://github.com/TES5Edit/TES5Edit/blob/dev-4.1.6/Core/wbDefinitionsTES5.pas)
(ARMA DNAM, NPC NAM7/PNAM, HDPT PNAM/HNAM). Partition metadata follows
[Niftools' NIF schema](https://github.com/niftools/nifxml/blob/develop/nif.xml).
Retail data and captures remain local.

Synthetic tests cover slot contention, preservation of armor on weapon changes,
script policies, explicit unequipping, hair extra parts, skin coverage, weight
interpolation/rejection, and draw/sheath save and stream continuation. These
verify engine behavior, not identical execution of retail or FNIS graphs.

Local Ralof traces resolve `Weapons\Iron\WarAxe.nif`, first to `WeaponAxe`, then
to `WEAPON` during retail `1hm_equip.hkx` playback. The trace explicitly records
`graphExecuted=0`: this is clip-event integration through the shared recovery
runtime and does not establish authored graph parity.

## Remaining acceptance and implementation gaps

- Execute admitted retail/FNIS graph branches rather than recovery selection.
  Local FNIS-generated output, an animation pack and its compatible skeleton
  have not been supplied for validation.
- Validate every weapon family, sex/race variant and custom attachment layout.
  Same-item dual wield, compound scabbards, mixed head/ear partitions and
  first-person equipment are not verified by the Ralof test.
- Complete equipment sound/event delivery to all attached scripts, enchantment
  presentation, and full instance-level item metadata. Those are not supplied
  by slot ownership alone.
- Sleeping outfits and arbitrary wardrobe-selection AI remain outside the
  implemented awake NPC presentation path.

Local capture command (optimized build, 768×432 logical / native-DPI framebuffer,
render scale 1): set `ODAI_ACTOR_WEAPON_DRAW=Ralof` alongside
`ODAI_CAPTURE_FOLLOW_ACTOR=Ralof` to request one draw after equipment initialization.
This opt-in probe logs clip markers and attachment changes; it does not mark
fallback execution as FNIS support.

## Latest local result (2026-09-10)

Debug and RelWithDebInfo builds succeed; all 45 Debug CTest cases pass, including
combat draw gating and automatic versus explicit draw ownership. The optimized
Ralof capture at 1536×864 shows a clean Stormcloak sleeve/body boundary and the
war axe in his hand. The resolved wardrobe no longer includes the duplicate
naked torso. No weight-morph rejection was reported in that capture.

Evidence is local under `captures/ralof-equipment-validation/`: `comparison.png`,
`ralof-drawn.png`, `retail-draw.log`, retail rig/wardrobe probes and `ctest.log`.
These results do not change the full-graph/FNIS acceptance status above.
