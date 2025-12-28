#pragma once

#include <cstddef>
#include <istream>
#include <memory>
#include <string>

namespace octotiger {

// Returns a binary input stream for the given relative dump path.
// The caller owns the returned stream.
std::unique_ptr<std::istream> open_dump_stream(std::string const& relative_path);

// Convenience helpers for the canonical dump filenames.
std::unique_ptr<std::istream> open_args_stream(std::size_t index);
std::unique_ptr<std::istream> open_outs_stream(std::size_t index);

} // namespace octotiger


