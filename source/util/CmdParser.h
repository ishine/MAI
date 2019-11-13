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

enum class CmdStatus {
    HELP, // print help info
    ERROR, // parse error
    SUCCESS, // parse success
};

#define ReaderFunction std::function<std::string(const std::string& valueStr, T& value)>

template<class T>
std::string defaultReader(const std::string& str, T& value) {
    std::istringstream ss(str);

    if (!(ss >> value && ss.eof())) {
        std::stringstream ss;
        ss << "value(" << str << ") cannot be parsed";
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
struct EnumReader {
public:
    EnumReader(const std::map<std::string, T>& values) : mValues(values) {
    }
    std::string operator()(const std::string& str, T& value) {
        std::string valueStr;
        std::string error = defaultReader<std::string>(str, valueStr);
        if (!error.empty()) {
            return error;
        }

        if (mValues.find(valueStr) == mValues.end()) {
            std::stringstream ss;
            ss << "value(" << value << ") must be one of {" << enumToString() << "}";
            return ss.str();
        }
        value = mValues[valueStr];
        return "";
    }

    EnumReader<T>& add(T v) {
        mValues.emplace_back(v);
        return *this;
    }
private:
    std::string enumToString() {
        std::stringstream ss;
        for (auto kv : mValues) {
            ss << kv.first << ", ";
        }
        return ss.str();
    }
private:
    std::map<std::string, T> mValues;
};

template<class T>
struct OneOfReader {
public:
    OneOfReader(const std::vector<T>& values) : mValues(values) {
    }
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

template<class T, class T2 = T>
struct ListReader {
public:
    ListReader(std::function<std::string(const std::string& valueStr, T2& value)> reader = defaultReader<T2>, const char& sep = ',')
        : mReader(reader), mSep(sep) {
    }

    std::string operator()(const std::string& str, std::vector<T>& value) {
        return strSplit(str, mSep, value);
    }

    std::string strSplit(const std::string& s, const char& sep, std::vector<T>& value) {
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, sep)) {
            T2 v;
            std::string error = mReader(item, v);
            if (!error.empty()) {
                return error;
            }
            value.push_back(v);
        }
        return "";
    }

private:
    std::function<std::string(const std::string& valueStr, T2& value)> mReader;
    const char& mSep;
};

class CmdParser {
public:
    inline CmdParser& add(const std::string& longName, const std::string& desc) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser(this, longName, 0, desc));
        mOptions[longName] = parser;
        return *this;
    }

    inline CmdParser& add(char shortName, const std::string& desc) {
        std::string name(1, shortName);
        if (mOptions.count(name)) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser(this, "", shortName, desc));
        mOptions[name] = parser;
        return *this;
    }

    inline CmdParser& add(const std::string& longName, char shortName, const std::string& desc) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }

        if (mOptions.count(std::string(1, shortName))) {
            MAI_ABORT("Repeated flag:%c", shortName);
            return *this;
        }
        std::shared_ptr<Parser> parser(new FlagParser(this, longName, shortName, desc));
        mOptions[std::string(1, shortName)] = parser;
        mOptions[longName] = parser;
        return *this;
    }

    template<class T>
    CmdParser& add(const std::string& longName, const std::string& desc,
            bool must, T def = T(), ReaderFunction valueReader = defaultReader<T>) {
        if (mOptions.count(longName)) {
            MAI_ABORT("Repeated flag:%s", longName.c_str());
            return *this;
        }
        std::shared_ptr<Parser> parser(new KeyValueParser<T>(this, longName, 0, desc, must, def, valueReader));
        mOptions[longName] = parser;
        return *this;
    }

    inline bool exist(const char& flag) const {
        return exist(std::string(1, flag));
    }

    inline bool exist(const std::string& name) const {
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

        const KeyValueParser<T>* parser = reinterpret_cast<const KeyValueParser<T>*>(it->second.get());
        MAI_CHECK(parser != NULL, "%s is not a k-v flag", name.c_str());
        return parser->read();
    }

    template<class T>
    const std::vector<T> getList(const std::string& name) const {
        auto it = mOptions.find(name);
        MAI_CHECK(it != mOptions.end(), "Unsupported flag:%s", name.c_str());

        const KeyValueParser<T>* parser = reinterpret_cast<const KeyValueParser<T>*>(it->second.get());
        MAI_CHECK(parser != NULL, "%s is not a k-v flag", name.c_str());
        return parser->readList();
    }

    template<class T>
    const T get(const char& flag) const {
        return get<T>(std::string(1, flag));
    }

    template<class T>
    const std::vector<T> getList(const char& flag) const {
        return getList<T>(std::string(1, flag));
    }

    inline std::stringstream& addErrorLog() {
        return mErrorStringStream;
    }

    inline std::string getErrorLog() {
        return mErrorStringStream.str();
    }

    inline std::string getHelpInfo() {
        std::stringstream ss;
        ss << mProgramName << std::endl << std::endl;
        ss << "Usage:" << std::endl;
        std::vector<std::string> processedKey;
        for(auto kv : mOptions) {
            if (std::find(processedKey.begin(), processedKey.end(), kv.second->longName()) == processedKey.end()
                    && std::find(processedKey.begin(), processedKey.end(), kv.second->shortName()) == processedKey.end()) {
                if (!(kv.second->shortName().empty())) {
                    processedKey.emplace_back(kv.second->shortName());
                }
                if (!(kv.second->longName().empty())) {
                    processedKey.emplace_back(kv.second->longName());
                }
                ss << kv.second->getHelpInfo() << std::endl;
            }
        }
        return ss.str();
    }

    CmdStatus parse(int argc, const char* const argv[]) {
        // add default help info
        if (mOptions.find("h") == mOptions.end() && mOptions.find("help") == mOptions.end()) {
            add("help", 'h', "Help Info");
        }
        if (argc < 1) {
            return CmdStatus::ERROR;
        }

        mProgramName = argv[0];// first data is program name
        if (argc > 1 && (!strncmp(argv[1], "-h", 2) || !strncmp(argv[1], "--help", 6))) {
            //printf(getHelpInfo().c_str());
            return CmdStatus::HELP;
        }

        for (int i = 1; i < argc; ++i) {
            if (strncmp(argv[i], "--", 2) == 0) { // long flag
                const char* p = strchr(argv[i] + 2, '=');
                if (p) {// long flag has a value
                    std::string name(argv[i] + 2, p);
                    std::string value(p+1);
                    auto it = mOptions.find(name);
                    if (it != mOptions.end()) {
                        if(!mOptions[name]->setValue(value)) {
                            continue;
                        }
                    } else {
                        addErrorLog() << "Cannot parse flag:" << name << std::endl;
                        continue;
                    }
                } else {// no value set for long flag
                    std::string name(argv[i] + 2);
                    auto it = mOptions.find(name);
                    if (it == mOptions.end()) {
                        addErrorLog() << "Cannot parse flag:" << name << std::endl;
                        continue;
                    }

                    if (it->second->needValue()) {
                        if (i + 1 >= argc) {
                            addErrorLog() << name << " miss value" << std::endl;
                            continue;
                        } else {
                            if (!it->second->setValue(argv[++i])) {
                                addErrorLog() << "Cannot set value(" << argv[i-1]
                                    << ") for flag(" << name << ")" << std::endl;
                                continue;
                            }
                        }
                    } else {
                        if (!it->second->set()) {
                            addErrorLog() << name << " need value" << std::endl;
                            continue;
                        }
                    }
                }
            } else if (strncmp(argv[i], "-", 1) == 0) { // short flag
                if (!argv[i][1]) {
                    continue;
                }
                char last = argv[i][1];
                for (int j = 2; argv[i][j]; j++) {
                    last = argv[i][j];
                    auto it = mOptions.find(std::string(1, argv[i][j - 1]));
                    if (it == mOptions.end()) {
                        addErrorLog() << "Cannot parse flag:" << argv[i][j-1] << std::endl;
                        continue;
                    }

                    it->second->set();
                }
                auto it = mOptions.find(std::string(1, last));
                if (it == mOptions.end()) {
                    addErrorLog() << "Cannot parse flag:" << last << std::endl;
                    continue;
                }

                if (it->second->needValue()) {
                    if (i + 1 < argc) {
                        if (!it->second->setValue(argv[i+1])) {
                            addErrorLog() << "Cannot set value(" << argv[i+1]
                                << ") for flag(" << last <<")"<< std::endl;
                        }
                    } else {
                        addErrorLog() << last << " miss value" << std::endl;
                        continue;
                    }
                } else {
                    it->second->set();
                }
            }
        }

        for (auto it = mOptions.begin(); it != mOptions.end(); ++it) {
            if (it->second->must() && !it->second->isSet()) {
                addErrorLog() << it->second->name() << " must be set" << std::endl;
                continue;
            }
        }

        if (!mErrorStringStream.str().empty()) {
            //ALOGE(mErrorStringStream.str().c_str());
            //printf(getHelpInfo().c_str());
            return CmdStatus::ERROR;
        }
        return CmdStatus::SUCCESS;
    }
private:
    class Parser {
    public:
        Parser(CmdParser* cmdParser) : mCmdParser(cmdParser) {
        }
        virtual ~Parser() {}
        virtual bool needValue() const = 0;
        virtual bool set() = 0;// for flag option
        virtual bool setValue(const std::string& value) = 0; // for key-value option
        virtual bool isSet() const = 0;
        virtual bool must() const = 0;
        virtual std::string shortName() const = 0;
        virtual std::string longName() const = 0;
        virtual std::string name() const = 0;
        virtual std::string getHelpInfo() const = 0;
        void addErrorLog(const std::string& errLog) {
            mCmdParser->addErrorLog() << errLog << std::endl;
        }

        std::stringstream& addErrorLog() {
            return mCmdParser->addErrorLog();
        }
    private:
        CmdParser* mCmdParser;
    };

    class FlagParser : public Parser {
    public:
        FlagParser(CmdParser* cmdParser, const std::string& longName,
                char shortName, const std::string& desc)
            : Parser(cmdParser), mIsSet(false),
            mLongName(longName), mDesc(desc) {
            if (shortName != 0) {
                mShortName = std::string(1, shortName);
            }

        }

        inline bool needValue() const {
            return false;
        }

        inline bool set() {
            if (mIsSet) {
                addErrorLog() << "Flag " << name() << " has been set" << std::endl;
                return false;
            }
            mIsSet = true;
            return true;
        }

        inline bool setValue(const std::string& value) {
            addErrorLog() << "Flag " << name() << " no need value" << std::endl;
            return false;
        }

        inline bool isSet() const {
            return mIsSet;
        }

        inline bool must() const {// FLAG no need to be set must
            return false;
        }

        inline std::string shortName() const {
            return mShortName;
        }

        inline std::string longName() const {
            return mLongName;
        }

        inline std::string name() const {
            return mLongName.empty() ? mShortName : mLongName;
        }

        inline std::string getHelpInfo() const {
            std::stringstream ss;
            if (!mShortName.empty() && !mLongName.empty()) {
                ss << "-" << mShortName << ", --" << mLongName;
            } else if (mShortName.empty()) {
                ss << "-" << mShortName;
            } else if (!mLongName.empty()) {
                ss << "--" << mLongName;
            }
            if (!mDesc.empty()) {
                ss << "\t\t" << mDesc;
            }
            return ss.str();
        }

    private:
        bool mIsSet;
        std::string mShortName;
        std::string mLongName;
        std::string mDesc;
    };

    template<class T>
    class KeyValueParser : public Parser {
    public:
        KeyValueParser(CmdParser* cmdParser, const std::string& longName, char shortName, const std::string& desc,
                bool must, T& def, ReaderFunction valueReader)
            : Parser(cmdParser), mMust(must), mIsSet(false),
            mLongName(longName), mDesc(desc), mReader(valueReader) {
            if (shortName != 0) {
                mShortName = std::string(1, shortName);
            }

            mValues.push_back(def);
        }

        KeyValueParser(CmdParser* cmdParser, const std::string& longName, char shortName, const std::string& desc,
                bool must, ReaderFunction valueReader)
            : Parser(cmdParser), mMust(must), mIsSet(false),
            mLongName(longName), mDesc(desc), mReader(valueReader) {
            if (shortName != 0) {
                mShortName = std::string(1, shortName);
            }
        }

        inline bool needValue() const {
            return true;
        }

        inline bool set() {
            addErrorLog() << "Need a value for flag:" << name() << std::endl;
            return false;
        }

        inline bool setValue(const std::string& value) {
            std::string r;
            if (mIsSet) {
                mValues.resize(mValues.size() + 1);
                r = mReader(value, mValues[mValues.size() - 1]);
            } else {
                r = mReader(value, mValues[0]);
            }
            if (r.empty()) {
                mIsSet = true;
                return true;
            } else {
                addErrorLog() << "Error: " << r << std::endl;
                return false;
            }
        }

        inline bool isSet() const {
            return mIsSet;
        }

        inline bool must() const {
            return mMust;
        }

        inline std::string shortName() const {
            return mShortName;
        }

        inline std::string longName() const {
            return mLongName;
        }

        inline std::string name() const {
            return mLongName.empty() ? mShortName : mLongName;
        }

        inline T read() const {
            return mValues[0];
        }

        inline std::vector<T> readList() const {
            return mValues;
        }

        inline std::string getHelpInfo() const {
            std::stringstream ss;
            if (!mShortName.empty() && !mLongName.empty()) {
                ss << "-" << mShortName << ", --" << mLongName;
            } else if (!mShortName.empty()) {
                ss << "-" << mShortName;
            } else if (!mLongName.empty()) {
                ss << "--" << mLongName;
            }
            ss << "\t\t";
            if (mMust) {
                ss <<"(required)";
            }
            if (!mDesc.empty()) {
                ss << mDesc;
            }
            return ss.str();
        }

    private:
        bool mMust;
        bool mIsSet;
        std::string mShortName;
        std::string mLongName;
        std::string mDesc;
        std::vector<T> mValues;
        ReaderFunction mReader;
    };

private:
    std::map<std::string, std::shared_ptr<Parser>> mOptions;
    std::string mProgramName;
    std::stringstream mErrorStringStream;
};
} //namespace MAI
