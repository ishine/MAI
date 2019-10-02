// Copyright 2019 MAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>// std::find
#include <string>
#include <cstring>
#include <sstream>
#include <functional>
#include <vector>
#include <map>
#include <memory>
#include "MAIUtil.h"

namespace MAI {

template<class T>
std::string defaultReader(const std::string& str, T& value) {
    std::istringstream ss(str);
    if (!(ss >> value && ss.eof())) {
        std::stringstream ss;
        ss << "value(" << value << ") cannot be converted to T";
        return ss.str();
    }
    return "";
}

template<class T>
struct RangeReader {
public:
    RangeReader(T low, T high) : mLow(low), mHigh(high) {}
    std::string operator()(const std::string& str, T& value) {
        std::string error = defaultReader<T>(str, value);
        if (!error.empty()) {
            return error;
        }

        if (!(value >= mLow && value <= mHigh)) {
            std::stringstream ss;
            ss << "value(" << value << ") must be in range[" << mLow << ", " << mHigh << "]";
            return ss.str();
        }
        return "";
    }
private:
    T mLow, mHigh;
};

template<class T>
struct OneOfReader {
public:
    std::string operator()(const std::string& str, T& value) {
        std::string error = defaultReader<T>(str, value);
        if (!error.empty()) {
            return error;
        }

        if (std::find(mValues.begin(), mValues.end(), value) == mValues.end()) {
            std::stringstream ss;
            ss << "value(" << value << ") must be one of {" << vectorToString(mValues) << "}";
            return ss.str();
        }
        return "";
    }

    OneOfReader<T>& add(T v) {
        mValues.emplace_back(v);
        return *this;
    }
private:
    std::vector<T> mValues;
};

template<class T>
struct ListReader {
public:
    ListReader(const std::string& sep = ",")
        : mSep(sep) {
    }

    std::string operator()(const std::string& str, std::vector<T>& value) {
        return strSplit(str, ',', value);
    }

    std::string strSplit(const std::string& s, const char& sep, std::vector<T>& value) {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, sep)) {
            T v;
            std::string error = defaultReader(item, v);
            if (!error.empty()) {
                return error;
            }
            value.push_back(v);
        }
        return "";
    }

private:
    const std::string& mSep;
};

class CmdParser {
public:
    CmdParser& add(const std::string& longName, const std::string& desc) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser(longName, 0, desc));
        mOptions[longName] = parser;
        return *this;
    }

    CmdParser& add(char shortName, const std::string& desc) {
        std::string name(1, shortName);
        if (mOptions.count(name)) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser("", shortName, desc));
        mOptions[name] = parser;
        return *this;
    }
    CmdParser& add(const std::string& longName, char shortName, const std::string& desc) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }

        if (mOptions.count(std::string(1, shortName))) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser(longName, shortName, desc));
        mOptions[std::string(1, shortName)] = parser;
        mOptions[longName] = parser;
        return *this;
    }

    template<class T>
    CmdParser& add(const std::string& longName, char shortName, const std::string& desc,
            bool must, T def, std::function<std::string(const std::string& valueStr, T& value)> valueReader = defaultReader<T>) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }

        if (mOptions.count(std::string(1, shortName))) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new KeyValueParser<T>(longName, shortName, desc, must, def, valueReader));
        mOptions[std::string(1, shortName)] = parser;
        mOptions[longName] = parser;
        return *this;
    }

    template<class T>
    CmdParser& add(char shortName, const std::string& desc,
            bool must, T def, std::function<std::string(const std::string& valueStr, T& value)> valueReader = defaultReader<T>) {
        if (mOptions.count(std::string(1, shortName))) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new KeyValueParser<T>("", shortName, desc, must, def, valueReader));
        mOptions[std::string(1, shortName)] = parser;
        return *this;
    }

    template<class T>
    CmdParser& add(const std::string& longName, const std::string& desc,
            bool must, T def, std::function<std::string(const std::string& valueStr, T& value)> valueReader = defaultReader<T>) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }
        std::shared_ptr<Parser> parser(new KeyValueParser<T>(longName, 0, desc, must, def, valueReader));
        mOptions[longName] = parser;
        return *this;
    }

    bool exist(const char& flag) const {
        return exist(std::string(1, flag));
    }

    bool exist(const std::string& name) const {
        if (mOptions.count(name) == 0) {
            return false;
        }
        if (mOptions.find(name)->second->isSet()) {
            return true;
        } else {
            return false;
        }
    }

    template<class T>
    const T get(const std::string& name) const {
        auto it = mOptions.find(name);
        MAI_CHECK(it != mOptions.end(), "Unsupported flag:%s", name.c_str());

        const KeyValueParser<T>* parser = dynamic_cast<const KeyValueParser<T>*>(it->second.get());
        MAI_CHECK(parser != NULL, "%s is not a k-v flag", name.c_str());
        return parser->read();
    }

    template<class T>
    const T get(const char& flag) const {
        return get<T>(std::string(1, flag));
    }

    void parse(int argc, const char* const argv[]) {
        if (argc < 1) {
            return;
        }

        mProgramName = argv[0];// first data is program name

        for (int i = 1; i < argc; ++i) {
            if (strncmp(argv[i], "--", 2) == 0) {
                const char* p = strchr(argv[i] + 2, '=');
                if (p) {
                    std::string name(argv[i] + 2, p);
                    std::string value(p+1);
                    auto it = mOptions.find(name);
                    if (it != mOptions.end()) {
                        if(!mOptions[name]->setValue(value)) {
                            MAI_ABORT("Cannot set value(%s) for flag(%s)", value.c_str(), name.c_str());
                        }
                    } else {
                        MAI_ABORT("Cannot parse flag:%s", name.c_str());
                        continue;
                    }
                } else {
                    std::string name(argv[i] + 2);
                    auto it = mOptions.find(name);
                    if (it == mOptions.end()) {
                        MAI_ABORT("Cannot parse flag:%s", name.c_str());
                        continue;
                    }

                    if (it->second->needValue()) {
                        if (i + 1 >= argc) {
                            MAI_ABORT("%s miss value", name.c_str());
                            continue;
                        } else {
                            if (!it->second->setValue(argv[++i])) {
                                MAI_ABORT("Cannot set value(%s) for flag(%s)", argv[++i], name.c_str());
                                continue;
                            }
                        }
                    } else {
                        if (!it->second->set()) {
                            MAI_ABORT("%s need value", name.c_str());
                            continue;
                        }
                    }
                }
            } else if (strncmp(argv[i], "-", 1) == 0) {
                if (!argv[i][1]) {
                    continue;
                }
                char last = argv[i][1];
                for (int j = 2; argv[i][j]; j++) {
                    last = argv[i][j];
                    auto it = mOptions.find(std::string(1, argv[i][j - 1]));
                    if (it == mOptions.end()) {
                        MAI_ABORT("Cannot parse flag:%c", argv[i][j-1]);
                        continue;
                    }

                    it->second->set();
                }
                auto it = mOptions.find(std::string(1, last));
                if (it == mOptions.end()) {
                    MAI_ABORT("Cannot parse flag:%c", last);
                    continue;
                }

                if (it->second->needValue()) {
                    if (i + 1 < argc) {
                        if (!it->second->setValue(argv[i+1])) {
                            MAI_ABORT("Cannot set value(%s) for flag(%c)", argv[i+1], last);
                        }
                    } else {
                        MAI_ABORT("%c miss value", last);
                        continue;
                    }
                } else {
                    it->second->set();
                }
            }
        }

        for (auto it = mOptions.begin(); it != mOptions.end(); ++it) {
            if (it->second->must() && !it->second->isSet()) {
                MAI_ABORT("%s must be set", it->second->name().c_str());
                continue;
            }
        }
    }
private:
    class Parser {
    public:
        virtual ~Parser() {}
        virtual bool needValue() const = 0;
        virtual bool set() = 0;// for flag option
        virtual bool setValue(const std::string& value) = 0; // for key-value option
        virtual bool isSet() const = 0;
        virtual bool must() const = 0;
        virtual char shortName() const = 0;
        virtual std::string longName() const = 0;
        virtual std::string name() const = 0;
    };

    class FlagParser : public Parser {
    public:
        FlagParser(const std::string& longName, char shortName, const std::string& desc)
            : Parser(), mIsSet(false), mShortName(shortName), mLongName(longName) {

        }

        bool needValue() const {
            return false;
        }

        bool set() {
            if (mIsSet) {
                MAI_ABORT("Flag %s has been set", name().c_str());
            }
            mIsSet = true;
            return true;
        }

        bool setValue(const std::string& value) {
            MAI_ABORT("Flag '%s' no need value", name().c_str());
            return false;
        }

        bool isSet() const {
            return mIsSet;
        }

        bool must() const {// FLAG no need to be set must
            return false;
        }

        char shortName() const {
            return mShortName;
        }
        std::string longName() const {
            return mLongName;
        }

        std::string name() const {
            return mLongName.empty() ? std::string(1, mShortName) : mLongName;
        }

    private:
        bool mIsSet;
        char mShortName;
        std::string mLongName;
    };

    template<class T>
    class KeyValueParser : public Parser {
    public:
        KeyValueParser(const std::string& longName, char shortName, const std::string& desc,
                bool must, T& def, std::function<std::string(const std::string&, T&)> valueReader)
            : Parser(), mMust(must), mIsSet(false), mShortName(shortName),
            mLongName(longName), mValue(def), mReader(valueReader) {

        }

        bool needValue() const {
            return true;
        }

        bool set() {
            MAI_ABORT("Need a value for flag:%s", name().c_str());
            return false;
        }

        bool setValue(const std::string& value) {
            if (mIsSet) {
                MAI_ABORT("Flag '%s' has been set", name().c_str());
            }
            std::string r = mReader(value, mValue);
            if (r.empty()) {
                mIsSet = true;
                return true;
            } else {
                MAI_ABORT("Error: %s", r.c_str());
                return false;
            }
        }

        bool isSet() const {
            return mIsSet;
        }

        bool must() const {
            return mMust;
        }

        char shortName() const {
            return mShortName;
        }

        std::string longName() const {
            return mLongName;
        }

        std::string name() const {
            return mLongName.empty() ? std::string(1, mShortName) : mLongName;
        }

        T read() const {
            return mValue;
        }
    private:
        bool mMust;
        bool mIsSet;
        char mShortName;
        std::string mLongName;
        T mValue;
        std::function<std::string(const std::string&, T&)> mReader;
    };

private:
    std::map<std::string, std::shared_ptr<Parser>> mOptions;
    std::string mProgramName;
};
} //namespace MAI
