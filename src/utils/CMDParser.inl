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
#include "CMDParser.hpp"

template <>
inline void cmdLineCleanUp<int>(int* t) {}

template <>
inline void cmdLineCleanUp<unsigned int>(unsigned int* t) {}

template <>
inline void cmdLineCleanUp<float>(float* t) {}

template <>
inline void cmdLineCleanUp<double>(double* t) {}

template <>
inline void cmdLineCleanUp<bool>(bool* t) {}

template <>
inline void cmdLineCleanUp<char*>(char** t)
{
	if (*t) free(*t);
	*t = nullptr;
}

template <>
inline int cmdLineInitialize<int>() { return 0; }

template <>
inline unsigned int cmdLineInitialize<unsigned int>() { return 0; }

template <>
inline float cmdLineInitialize<float>() { return 0.f; }

template <>
inline double cmdLineInitialize<double>() { return 0.; }

template <>
inline bool cmdLineInitialize<bool>() { return false; }

template <>
inline char* cmdLineInitialize<char*>() { return nullptr; }

template <>
inline int cmdLineCopy(int t) { return t; }

template <>
inline unsigned int cmdLineCopy(unsigned int t) { return t; }

template <>
inline float cmdLineCopy(float t) { return t; }

template <>
inline double cmdLineCopy(double t) { return t; }

template <>
inline bool cmdLineCopy(bool t) { return t; }

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
template <>
char* cmdLineCopy(char* t)
{
	return _strdup(t);
}
#else  // !WIN
template <>
inline char* cmdLineCopy(char* t)
{
	return strdup(t);
}
#endif // WIN

template <>
inline int cmdLineStringToT(const char* str) { return atoi(str); }

template <>
inline unsigned int cmdLineStringToT(const char* str) { return strtoul(str, nullptr, 10); }

template <>
inline float cmdLineStringToT(const char* str) { return float(atof(str)); }

template <>
inline double cmdLineStringToT(const char* str) { return double(atof(str)); }

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
template <>
inline char* cmdLineStringToT(const char* str)
{
	return _strdup(str);
}
#else  // !WIN
template <>
inline char* cmdLineStringToT(const char* str)
{
	return strdup(str);
}
#endif // WIN32 || _WIN64

/////////////////////
// ParserInterface //
/////////////////////
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
inline ParserInterface::ParserInterface(const char* name) : set(false)
{
	this->name = _strdup(name);
}
#else  // !WIN
inline ParserInterface::ParserInterface(const char* name) : set(false)
{
	this->name = strdup(name);
}
#endif // WIN

inline ParserInterface::~ParserInterface()
{
	if (name) free(name);
	name = nullptr;
}

inline int ParserInterface::read(int argc, char** argv)
{
	set = true;
	return 0;
}

//////////////////////
// CmdLineParameter //
//////////////////////
template <typename T>
CmdLineParameter<T>::CmdLineParameter(const char* name) : ParserInterface(name)
{
	value = cmdLineInitialize<T>();
}

template <typename T>
CmdLineParameter<T>::CmdLineParameter(const char* name, T v) : ParserInterface(name)
{
	value = cmdLineCopy<T>(v);
}

template <typename T>
CmdLineParameter<T>::~CmdLineParameter()
{
	cmdLineCleanUp(&value);
}

template <typename T>
inline int CmdLineParameter<T>::read(int argc, char** argv)
{
	if (argc > 0)
	{
		cmdLineCleanUp<T>(&value);
		value = cmdLineStringToT<T>(argv[0]);
		set = true;
		return 1;
	}
	else
		return 0;
}

template <>
inline int CmdLineParameter<bool>::read(int argc, char** argv)
{
	cmdLineCleanUp<bool>(&value);
	set = value = true;
	return 0;
}

inline void cmdLineParse(int argc, char** argv, ParserInterface** params)
{
	while (argc > 0)
	{
		if (argv[0][0] == '-' && argv[0][1] == '-')
		{
			ParserInterface* readable = nullptr;
			for (int i = 0; params[i] != nullptr && readable == nullptr; i++)
			{
				if (!strcasecmp(params[i]->name, argv[0] + 2))
					readable = params[i];
				if (i == argc)
					break;
			}
			if (readable)
			{
				/*{set = true; return 1;}*/
				int j = readable->read(argc - 1, argv + 1);
				argv += j, argc -= j;
			}
			else
			{
				printf("Invalid option: %s, options should like:\n", argv[0]);
				for (int i = 0; params[i] != nullptr; i++)
					fprintf(stderr, "\t--%s\n", params[i]->name);
			}
		}
		else
		{
			printf("Parameter name should be of the form '--%s value' or '--%s(%s must be of a bool type in this form)'\n", argv[0], argv[0], argv[0]);
		}
		++argv, --argc;
	}
}
