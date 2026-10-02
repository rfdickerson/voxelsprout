#pragma once

#include <string>

namespace odai::bethesda {

inline std::string normalizedEditorId(std::string value) {
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

}  // namespace odai::bethesda
