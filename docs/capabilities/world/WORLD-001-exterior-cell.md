# WORLD-001: Exterior Cell

## Goal

Iridius can represent a Morrowind exterior cell as a runtime world object.

## Required behavior

Given an exterior cell at grid coordinates (x, y):

- the cell can be created from loaded content data
- its grid coordinates are preserved
- its world-space bounds are known
- references belonging to that cell can be associated with it
- terrain belonging to the cell can be associated with it
- the cell can exist without requiring it to be rendered

## Architectural constraint

World state must not depend on renderer objects.

The renderer may consume world state, but the world layer must remain
usable when running headless.

## Out of scope

- streaming
- rendering
- physics
- NPC AI
- save/load