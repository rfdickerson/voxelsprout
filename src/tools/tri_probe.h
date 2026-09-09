#pragma once
#include <filesystem>
#include <string>
int probeTriMorphs(const std::filesystem::path &source, const std::string &path,
                   bool profile);
