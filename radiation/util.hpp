#pragma once

#include <exception>
#include <sstream>
#include <string>

template <typename... Ts>
std::string tprintf(char const* format, Ts... vs);

template <typename E = std::runtime_error, typename... Ts>
inline std::exception formatted_exception(Ts... vs)
{
    auto r = tprintf(std::forward<Ts>(vs)...);
    return E{r};
}

inline std::stringstream& sstprintf(std::stringstream& ss, char const* format)
{
    ss << format;
    return ss;
}

template <typename T, typename... Targs>
inline std::stringstream& sstprintf(
    std::stringstream& ss, char const* format, T value, Targs... Fargs)
{
    for (; *format != '\0'; ++format)
    {
        if (*format == '%')
        {
            ss << value;
            tprintf(format + 1, Fargs...);
            return ss;
        }
        ss << *format;
    }
    return ss;
}

template <typename... Ts>
inline std::string tprintf(char const* format, Ts... vs)
{
    std::stringstream ss;
    sstprintf(ss, format, std::forward<Ts>(vs)...);
    return ss.str();
}
