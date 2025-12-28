#include "utils/dump_reader.hpp"

#include "config.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <zstd.h>

namespace octotiger {
namespace {

constexpr char const* dump_prefix = "octotiger-radiation-";
constexpr std::uint32_t pack_magic = 0x4f44504b; // 'ODPK'
constexpr std::uint32_t pack_version = 2;

enum class file_kind : std::size_t
{
    args = 0,
    outs = 1
};

struct pack_header
{
    std::uint32_t magic = pack_magic;
    std::uint32_t version = pack_version;
    std::uint32_t entry_count = 0;
    std::uint32_t reserved = 0;
};

struct pack_entry
{
    std::uint64_t offsets[2] = {0, 0};
    std::uint32_t compressed_sizes[2] = {0, 0};
    std::uint32_t raw_sizes[2] = {0, 0};
};

std::string build_filename(std::size_t index, char const* extension)
{
    std::string name{dump_prefix};
    name += std::to_string(index);
    name += extension;
    return name;
}

std::size_t parse_case_index(std::string const& filename)
{
    auto dash = filename.find_last_of('-');
    auto dot = filename.find_last_of('.');
    if (dash == std::string::npos || dot == std::string::npos ||
        dot <= dash + 1)
    {
        throw formatted_exception(
            "error: malformed dump entry name \"%\"", filename.c_str());
    }
    auto const number = filename.substr(dash + 1, dot - dash - 1);
    return static_cast<std::size_t>(std::stoul(number));
}

bool has_suffix(std::string const& name, char const* suffix)
{
    auto const suffix_len = std::strlen(suffix);
    if (name.size() < suffix_len)
    {
        return false;
    }
    return std::equal(
        suffix, suffix + suffix_len, name.end() - static_cast<long>(suffix_len));
}

class membuf : public std::streambuf
{
public:
    membuf()
    {
        reset(nullptr, 0);
    }

    void reset(const char* base, std::size_t size)
    {
        auto* begin = const_cast<char*>((size == 0) ? &dummy_ : base);
        setg(begin, begin, begin + size);
    }

private:
    static char dummy_;
};

char membuf::dummy_ = 0;

class memory_istream : public std::istream
{
public:
    explicit memory_istream(std::vector<char>&& buffer)
      : std::istream(nullptr)
      , buffer_(std::move(buffer))
    {
        buf_.reset(
            buffer_.empty() ? nullptr : buffer_.data(), buffer_.size());
        this->rdbuf(&buf_);
    }

private:
    std::vector<char> buffer_;
    membuf buf_;
};

class dump_pack
{
public:
    explicit dump_pack(std::string pack_path)
      : pack_path_{std::move(pack_path)}
    {
        if (!load_pack())
        {
            throw formatted_exception(
                "error: dump pack \"%\" is missing. Re-run CMake to fetch arg-dumps.pack.zst.",
                pack_path_.c_str());
        }
    }

    ~dump_pack()
    {
        if (pack_file_ != nullptr)
        {
            std::fclose(pack_file_);
        }
    }

    std::vector<char> extract(std::string const& relative_path)
    {
        auto const kind = classify(relative_path);
        auto const index = parse_case_index(relative_path);
        std::lock_guard<std::mutex> lock(mutex_);
        return read_entry(index, kind);
    }

private:
    file_kind classify(std::string const& name) const
    {
        if (has_suffix(name, ".args"))
        {
            return file_kind::args;
        }
        if (has_suffix(name, ".outs"))
        {
            return file_kind::outs;
        }
        throw formatted_exception(
            "error: unsupported dump entry \"%\"", name.c_str());
    }

    bool load_pack()
    {
        pack_file_ = std::fopen(pack_path_.c_str(), "rb");
        if (!pack_file_)
        {
            return false;
        }

        pack_header header{};
        if (std::fread(&header, sizeof(header), 1, pack_file_) != 1)
        {
            std::fclose(pack_file_);
            pack_file_ = nullptr;
            return false;
        }

        if (header.magic != pack_magic || header.version != pack_version ||
            header.entry_count != OCTORAD_DUMP_COUNT)
        {
            std::fclose(pack_file_);
            pack_file_ = nullptr;
            return false;
        }

        entries_.assign(header.entry_count, {});
        if (std::fread(entries_.data(), sizeof(pack_entry), entries_.size(),
                pack_file_) != entries_.size())
        {
            std::fclose(pack_file_);
            pack_file_ = nullptr;
            return false;
        }

        return true;
    }

    std::vector<char> read_entry(std::size_t index, file_kind kind)
    {
        if (index >= entries_.size())
        {
            throw formatted_exception(
                "error: request for out-of-range case %", index);
        }

        auto const kind_index = static_cast<std::size_t>(kind);
        auto const& rec = entries_[index];
        auto const raw_size = rec.raw_sizes[kind_index];
        auto const comp_size = rec.compressed_sizes[kind_index];
        auto const offset = rec.offsets[kind_index];

        std::vector<char> output(raw_size);
        if (raw_size == 0)
        {
            return output;
        }

        std::vector<char> compressed(comp_size);
        if (comp_size > 0)
        {
            std::fseek(pack_file_, static_cast<long>(offset), SEEK_SET);
            if (std::fread(compressed.data(), 1, comp_size, pack_file_) !=
                comp_size)
            {
                throw formatted_exception(
                    "error: unable to read case %", index);
            }
        }

        std::size_t result = ZSTD_decompress(
            output.data(), output.size(), compressed.data(), comp_size);
        if (ZSTD_isError(result) || result != output.size())
        {
            throw formatted_exception(
                "error: corrupted dump data for case %", index);
        }
        return output;
    }

    std::string pack_path_;
    std::vector<pack_entry> entries_;
    FILE* pack_file_{nullptr};
    std::mutex mutex_;
};

} // namespace

std::unique_ptr<std::istream> open_dump_stream(
    std::string const& relative_path)
{
    static std::unique_ptr<dump_pack> pack;
    if (!pack)
    {
        std::string pack_path;
        if (char const* env_path = std::getenv("OCTORAD_DUMP_PACK"))
        {
            pack_path = env_path;
        }
        else
        {
            pack_path = std::string(OCTORAD_DUMPS_PACK);
        }
        pack = std::make_unique<dump_pack>(pack_path);
    }
    auto buffer = pack->extract(relative_path);
    return std::make_unique<memory_istream>(std::move(buffer));
}

std::unique_ptr<std::istream> open_args_stream(std::size_t index)
{
    return open_dump_stream(build_filename(index, ".args"));
}

std::unique_ptr<std::istream> open_outs_stream(std::size_t index)
{
    return open_dump_stream(build_filename(index, ".outs"));
}

} // namespace octotiger
