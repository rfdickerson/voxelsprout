#pragma once
// Source compatibility for the original runtime namespaces.
namespace odai::games::bethesda {}
namespace odai::games { namespace newvegas = bethesda; }
namespace odai { namespace newvegas = games::bethesda; }
