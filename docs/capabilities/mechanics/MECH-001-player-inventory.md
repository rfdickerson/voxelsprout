# MECH-001: Player Inventory

Status: Planned

## Goal

Provide the player with a basic Morrowind-style inventory that can be opened through the game UI, displays the items currently owned by the player, and allows an item to be dropped into the world.

This capability establishes the minimum inventory lifecycle:

    world item
        ↓
    player inventory
        ↓
    inventory UI
        ↓
    dropped world item

More advanced inventory behavior should be implemented as separate capabilities.

## Dependencies

- HARNESS-001: Headless Engine Test Runner
- Player entity/state system
- World object/entity representation
- Basic UI framework
- Item/content record loading sufficient to identify inventory objects

## Required Behavior

### MECH-001-A: Player inventory state

The player has an inventory capable of containing item instances.

At minimum, each inventory entry must preserve enough information to:

- identify the item's content record/type
- determine its quantity
- display it in the inventory UI
- instantiate the corresponding item in the world if dropped

The exact internal representation is an implementation decision.

### MECH-001-B: Open inventory UI

The player can open and close the inventory interface using the normal game input/UI flow.

Opening the inventory displays the player's current inventory.

The capability must not introduce a test-only pathway for opening the inventory.

### MECH-001-C: Display owned items

The inventory UI displays items currently owned by the player.

At minimum, each visible inventory entry must provide enough information for the player to distinguish the item.

For this initial capability, the UI should support:

- item name
- quantity when greater than one

Icons, item statistics, categories, tooltips, sorting, filtering, and equipment state are outside the initial scope unless already provided naturally by existing systems.

### MECH-001-D: Drop an item

The player can select an item from inventory and drop it.

Dropping an item must:

1. remove the appropriate quantity from the player's inventory
2. create or restore a corresponding world object
3. place that object at a reasonable location near the player
4. preserve the item's identity and relevant instance state

If the inventory contains a stack, dropping one item should reduce the stack rather than necessarily removing the entire entry.

The exact interaction used to select quantity may be deferred if no quantity-selection UI exists yet.

### MECH-001-E: Inventory/world consistency

An item must not simultaneously exist as both:

- an owned inventory item, and
- the same active world instance

Moving an item between world and inventory representation must preserve ownership/state consistently.

### MECH-001-F: Reopen inventory

After dropping an item:

- reopening or refreshing the inventory must reflect the new quantity
- the dropped item must remain present in the world

## Initial Vertical Slice

Create a synthetic test scenario containing:

- player
- at least two distinct inventory item types
- one item with quantity greater than one

Example initial state:

    Iron Dagger       x1
    Restore Health    x3

The scenario should demonstrate:

1. player starts with the expected inventory
2. inventory UI can be opened
3. both item types are represented
4. player drops one Restore Health item
5. inventory quantity becomes 2
6. one corresponding item exists in the world near the player
7. closing and reopening inventory still shows quantity 2

## Automated Verification

Using the headless/scenario testing infrastructure, verify the underlying inventory behavior independently of visual presentation.

At minimum:

### INV-001: Initial inventory

Given:

    Iron Dagger       x1
    Restore Health    x3

Assert:

    inventory contains Iron Dagger
    Iron Dagger quantity == 1
    inventory contains Restore Health
    Restore Health quantity == 3

### INV-002: Drop single item

Drop one Restore Health item.

Assert:

    inventory Restore Health quantity == 2
    world contains one newly dropped Restore Health instance

### INV-003: Drop final item

Given:

    Iron Dagger x1

Drop the Iron Dagger.

Assert:

    inventory no longer contains Iron Dagger
    world contains the dropped Iron Dagger

### INV-004: State preservation

Drop an item with instance-specific state, if such state already exists in the current item model.

Assert that relevant state survives the inventory-to-world transition.

Do not introduce an artificial state system solely for this test.

## UI Verification

The capability should also include an automated or deterministic UI-level verification where practical.

Verify that:

1. inventory can be opened through the normal UI/input system
2. owned items appear in the inventory view
3. displayed quantities match inventory state
4. dropping an item causes the UI to reflect the updated inventory

Visual regression testing is not required for this capability unless the existing harness already supports it.

## Architectural Constraints

- Inventory state must belong to gameplay/world state, not to the UI.
- The UI should observe/manipulate inventory through appropriate engine interfaces.
- Do not make UI widgets the authoritative source of item ownership.
- Avoid duplicating world-item and inventory-item state unnecessarily.
- Preserve a clean path for future save/load support.
- Do not hard-code Morrowind item types into generic UI infrastructure.
- Prefer existing Iridius entity/content abstractions over introducing parallel inventory-specific representations.

## Out of Scope

The following are intentionally deferred:

- picking items up from the world
- containers
- NPC inventory
- trading
- equipment slots
- equipping/unequipping
- armor and weapon statistics
- encumbrance
- inventory categories/tabs
- item icons
- tooltips
- item value
- item condition
- stealing/ownership/crime
- drag-and-drop between containers
- stack-splitting UI
- barter
- save/load persistence unless required by existing architecture

These should be separate capabilities.

## Future Capabilities

Likely follow-up capabilities include:

- MECH-003: Pick Up World Items
- MECH-004: Equipment and Equipped Items
- MECH-005: Containers
- MECH-006: NPC Inventory
- MECH-007: Encumbrance
- UI-INV-001: Inventory Icons and Tooltips
- UI-INV-002: Drag-and-Drop Inventory Interaction
- MECH-TRADE-001: Barter and Trading

## Definition of Done

MECH-001 is Implemented when:

- player inventory state exists
- player can open and close the inventory UI
- current inventory contents are displayed
- quantities are represented correctly
- player can drop an item from inventory
- dropping updates inventory state
- dropped item appears in the world
- inventory/world state remains consistent
- automated inventory scenarios pass
- relevant UI verification passes
- existing test suite passes
- no unrelated regressions are introduced
- docs/PARITY.md is updated to mark MECH-001 Implemented
