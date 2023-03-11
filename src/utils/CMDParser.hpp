/*
 * @Author: WangLei
 * @authorEmail: leiw1006@gmail.com
 * @Date: 2023-01-29 12:31:31
 * @LastEditors: WangLei
 * @LastEditTime: 2023-01-30 10:50:00
 * @FilePath: 
 * @Description:
 */
#pragma once
#include <cstring>
#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// 执行不区分大小写的字符串比较
int strcasecmp(const char *c1, const char *c2) { return _stricmp(c1, c2); }
#endif // WIN

template <typename T>
inline void cmdLineCleanUp(T* t);

template <typename T>
inline T cmdLineInitialize();

template <typename T>
inline T cmdLineCopy(T t);

template <typename T>
inline T cmdLineStringToT(const char* str);

class ParserInterface
{
public:
    bool set;
    char *name;

    ParserInterface(const char *name);

    virtual ~ParserInterface();

    virtual int read(int argc, char **argv);
};

template <typename T>
class CmdLineParameter : public ParserInterface
{
public:
    T value;

    CmdLineParameter(const char *name);

    CmdLineParameter(const char *name, T v);

    ~CmdLineParameter() override;

    inline int read(int argc, char **argv) override;
};

inline void cmdLineParse(int argc, char** argv, ParserInterface** params);

#include "CMDParser.inl"