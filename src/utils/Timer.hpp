/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-01-28 11:15:58
 * @LastEditors: WangLei
 * @LastEditTime: 2023-01-28 23:34:24
 * @FilePath:
 * @Description:
 */

// cpu count timer

#include <iostream>

class TimerInterface
{
public:
    TimerInterface() = default;
    virtual ~TimerInterface() {}

public:
    // Start time measurement
    virtual void start() = 0;

    // Stop time measurement
    virtual void stop() = 0;

    // Reset time counters to zero
    virtual void reset() = 0;

    // Time in msec. after start. If the stop watch is still running (i.e. there
    // was no call to stop()) then the elapsed time is returned, otherwise the
    // time between the last start() and stop call is returned
    virtual double getTime() = 0;

    // Mean time to date based on the number of times the stopwatch has been
    // _stopped_ (ie finished sessions) and the current total time
    virtual double getAverageTime() = 0;
};

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

class WinTimer : public TimerInterface
{
public:
    WinTimer() : start_time(), end_time(),
                 diff_time(0.0), total_time(0.0),
                 isRunning(false), sessions(0),
                 freq(0), isSetFreq(false)
    {
        if (!isSetFreq)
        {
            isSetFreq = true;
            // get the tick frequency from the OS
            LARGE_INTEGER temp;
            freq = (double)(QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&temp)) / 1000.0);
        }
    }
    ~WinTimer() {}

public:
    inline void start();

    inline void stop();

    inline void reset();

    inline double getTime();

    inline double getAverageTime();

private:
    LARGE_INTEGER start_time;

    LARGE_INTEGER end_time;

    double diff_time;

    double total_time;

    // flag if the stop watch is running
    bool isRunning;

    // Number of times clock has been started and stopped to allow averaging
    int sessions;

    // tick frequency
    double freq;

    // flag if the frequency has been set
    bool isSetFreq;
};

inline void WinTimer::start()
{
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
    isRunning = true;
}

inline void WinTimer::stop()
{
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&end_time));
    diff_time = (double)(((double)(end_time.QuadPart) -
                          (double)(start_time.QuadPart)) /
                         freq);
    total_time += diff_time;
    sessions++;
    isRunning = false;
}

inline void WinTimer::reset()
{
    diff_time = 0.;
    total_time = 0.;
    sessions = 0;
    /* Does not change the timer running state but does recapture this point
       in time as the current start time if it is running. */
    if (isRunning)
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
}

inline double WinTimer::getTime()
{
    double retval = total_time;
    if (isRunning)
    {
        LARGE_INTEGER temp;
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&temp));
        retval += (double)(((double)(temp.QuadPart) -
                            (double)(start_time.QuadPart)) /
                           freq);
    }
    return retval;
}

inline double WinTimer::getAverageTime()
{
    /* flag 'isRunning' is false when call the function */
    return (sessions > 0) ? (total_time / sessions) : 0.0;
}

#else

#include <sys/time.h>
#include <ctime>

class LinuxTimer : public TimerInterface
{
public:
    LinuxTimer() : start_time(), end_time(),
                   diff_time(0.0), total_time(0.0),
                   isRunning(false), sessions(0),
                   freq(0), isSetFreq(false)
    {
        if (!isSetFreq)
        {
            isSetFreq = true;
            // get the tick frequency from the OS
            LARGE_INTEGER temp;
            freq = (double)(QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&temp)) / 1000.0);
        }
    }
    ~LinuxTimer() {}

public:
    inline void start();

    inline void stop();

    inline void reset();

    inline double getTime();

    inline double getAverageTime();

private:
    inline double getDiffTime();

private:
    struct time_val start_time;

    double diff_time;

    double total_time;

    // flag if the stop watch is running
    bool isRunning;

    // Number of times clock has been started and stopped to allow averaging
    int sessions;
};

inline void LinuxTimer::start()
{
    gettimeofday(&start_time, 0);
    isRunning = true;
}

inline void LinuxTimer::stop()
{
    diff_time = getDiffTime();
    total_time += diff_time;
    sessions++;
    isRunning = false;
}

inline void LinuxTimer::reset()
{
    diff_time = 0.;
    total_time = 0.;
    sessions = 0;
    /* Does not change the timer running state but does recapture this point
       in time as the current start time if it is running. */
    if (isRunning)
        gettimeofday(&start_time, 0);
}

inline double LinuxTimer::getTime()
{
    double retval = total_time;
    if (isRunning)
        retval += getDiffTime();
    return retval;
}

inline double LinuxTimer::getAverageTime()
{
    /* flag 'isRunning' is false when call the function */
    return (sessions > 0) ? (total_time / sessions) : 0.0;
}

inline double StopWatchLinux::getDiffTime()
{
    struct timeval t_time;
    gettimeofday(&t_time, 0);

    // time difference in milli-seconds
    return (double)(1e3 * (t_time.tv_sec - start_time.tv_sec) +
                    (1e-3 * (t_time.tv_usec - start_time.tv_usec)));
}

#endif // WIN32

inline bool createTimer(TimerInterface **timer_interface)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    *timer_interface = reinterpret_cast<TimerInterface *>(new WinTimer());
#else
    *timer_interface = reinterpret_cast<TimerInterface *>(new LinuxTimer());
#endif
    return (*timer_interface) ? true : false;
}

inline bool deleteTimer(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        delete *timer_interface;
        *timer_interface = nullptr;
        return true;
    }
    else
    {
        printf("Can not DELETE a null pointer!\n");
        return false;
    }
}

inline bool startTimer(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        (*timer_interface)->start();
        return true;
    }
    else
    {
        printf("Can not START a null pointer!\n");
        return false;
    }
}

inline bool stopTimer(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        (*timer_interface)->stop();
        return true;
    }
    else
    {
        printf("Can not STOP a null pointer!\n");
        return false;
    }
}

inline bool resetTimer(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        (*timer_interface)->reset();
        return true;
    }
    else
    {
        printf("Can not RESET a null pointer!\n");
        return false;
    }
}

inline double getTimerValue(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        return (*timer_interface)->getTime();
    }
    else
    {
        printf("Can not GET TIME from a null pointer!\n");
        return 0.0;
    }
}

inline double getAverageTimerValue(TimerInterface **timer_interface)
{
    if (*timer_interface)
    {
        return (*timer_interface)->getAverageTime();
    }
    else
    {
        printf("Can not GET AVERAGE TIME from a null pointer!\n");
        return 0.0;
    }
}