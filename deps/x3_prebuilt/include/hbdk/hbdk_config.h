
// clang-format off
// @formatter:off
//===----- hbdk_config.h - Common flags for hbdk ---*- C -*-===//
// Automatically generated. DO NOT EDIT
//
//                     The HBDK Compiler Infrastructure
//
// This file is subject to the terms and conditions defined in file
// 'LICENSE.txt', which is part of this source code package.
//
//===-----------------------------------------------------------------------===//

#ifndef HBDK_CONFIG_H_
#define HBDK_CONFIG_H_
#pragma once

#define HBRT_VERSION_MAJOR 3U
#define HBRT_VERSION_MINOR 10U
#define HBRT_VERSION_PATCH 4U

#define HBDK_COMPILER_VERSION_MAJOR 3U
#define HBDK_COMPILER_VERSION_MINOR 15U
#define HBDK_COMPILER_VERSION_PATCH 4U

#define COMPILE_X2 1
#define COMPILE_X2A 1
#define COMPILE_X3 1


#define HBRT_INTERNAL_STR(x) #x

#ifdef __GNUC__
#define HBRT_PRAGMA_WARN(msg) _Pragma(HBRT_INTERNAL_STR(GCC warning msg))
#else
#define HBRT_PRAGMA_WARN(msg)
#endif

#ifdef __GNUC__
#if (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 1) // deprecated is supported since gcc 4.1
#if (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5) // deprecated with message is supported since gcc 4.5
#define HBRT_DEPRECATED(msg) __attribute__ ((deprecated(msg)))
#define HBRT_DEPRECATED_NAME(new_name, old_name, version_since) \
__attribute__ ((deprecated(#old_name " has been superseded by " #new_name " since " #version_since ", and will be removed in a future release")))
#else
#define HBRT_DEPRECATED(msg) __attribute__ ((deprecated))
#define HBRT_DEPRECATED_NAME(new_name, old_name, version_since) \
__attribute__ ((deprecated))
#endif
#else
#define HBRT_DEPRECATED(msg)
#define HBRT_DEPRECATED_NAME(new_name, old_name, version_since)
#endif
#else
#define HBRT_DEPRECATED(msg)
#define HBRT_DEPRECATED_NAME(new_name, old_name, version_since)
#endif

#if defined(HBDK_WHOLE_PROGRAM)
#define HBDK_PUBLIC __attribute__((__visibility__("default"))) __attribute__((externally_visible)) 
#elif defined(_MSC_VER)
#define HBDK_PUBLIC __declspec(dllexport)
#elif defined(__GNUC__)
#define HBDK_PUBLIC __attribute__((__visibility__("default")))
#else
#define HBDK_PUBLIC
#endif

#endif // HBDK_CONFIG_H_

