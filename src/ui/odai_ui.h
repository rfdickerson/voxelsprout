#pragma once

// Umbrella header for the entire odai_ui library.
// Include this for a new project; include individual headers for faster builds.

// Core
#include "ui/animation.h"
#include "ui/cached_rich_text.h"
#include "ui/font.h"
#include "ui/icon_atlas.h"
#include "ui/resource_style.h"
#include "ui/rich_text.h"
#include "ui/tooltip.h"
#include "ui/ui_context.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_input.h"
#include "ui/ui_text_util.h"
#include "ui/ui_types.h"
#include "ui/widget.h"

// Theming
#include "ui/theme/ui_theme.h"

// Data binding & declarative documents
#include "ui/document/ui_binding.h"
#include "ui/document/ui_document.h"
#include "ui/document/ui_hot_reload.h"

// Widgets — primitives
#include "ui/widgets/button.h"
#include "ui/widgets/context_menu.h"
#include "ui/widgets/dropdown.h"
#include "ui/widgets/icon_button.h"
#include "ui/widgets/image.h"
#include "ui/widgets/label.h"
#include "ui/widgets/modal.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/progress_bar.h"
#include "ui/widgets/radio_button.h"
#include "ui/widgets/repeater.h"
#include "ui/widgets/rich_text_view.h"
#include "ui/widgets/scroll_view.h"
#include "ui/widgets/slider.h"
#include "ui/widgets/spacer.h"
#include "ui/widgets/spinner.h"
#include "ui/widgets/stack_layout.h"
#include "ui/widgets/stat_badge.h"
#include "ui/widgets/tab_bar.h"
#include "ui/widgets/text_box.h"
#include "ui/widgets/toast.h"
#include "ui/widgets/toggle.h"
#include "ui/widgets/toolbar.h"
#include "ui/widgets/window.h"

// Widgets — charts
#include "ui/widgets/donut_chart.h"
#include "ui/widgets/line_chart.h"

// Game-agnostic panels
#include "ui/widgets/advisors_panel.h"
#include "ui/widgets/build_queue_panel.h"
#include "ui/widgets/event_tracker_panel.h"
#include "ui/widgets/faction_panel.h"
#include "ui/widgets/grid_picker_panel.h"
#include "ui/widgets/minimap_panel.h"
#include "ui/widgets/notable_entity_panel.h"
#include "ui/widgets/research_panel.h"
#include "ui/widgets/resource_bar_panel.h"
#include "ui/widgets/selection_inspector_panel.h"
#include "ui/widgets/sim_controls_panel.h"

// Genre kits (convenient sub-selections)
#include "ui/kits/city_builder_kit.h"
#include "ui/kits/colony_sim_kit.h"
#include "ui/kits/strategy_4x_kit.h"
