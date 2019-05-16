#pragma once

#include <chrono>
#include <cstddef>

template <typename T, typename R = std::milli>
struct scoped_timer;

template <typename T, std::intmax_t N, std::intmax_t D>
struct scoped_timer<T, std::ratio<N, D>>
{
    scoped_timer(T& r)
      : start_timepoint(std::chrono::high_resolution_clock::now())
      , value{r}
    {
    }
    ~scoped_timer()
    {
        value = std::chrono::duration<T, std::ratio<N, D>>(
            std::chrono::high_resolution_clock::now() - start_timepoint)
                    .count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_timepoint;
    T& value;
};
