#pragma once

#include <chrono>

template <typename T>
struct scoped_timer
{
    scoped_timer(T& r)
      : start_timepoint(std::chrono::high_resolution_clock::now())
      , value{r}
    {
    }
    ~scoped_timer()
    {
        value = std::chrono::duration<T, std::ratio<1, 1000000000>>(
            std::chrono::high_resolution_clock::now() - start_timepoint)
                    .count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_timepoint;
    T& value;
};