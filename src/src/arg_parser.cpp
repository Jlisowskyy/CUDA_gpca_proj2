/* internal includes */
#include <arg_parser.hpp>
#include <tester.hpp>
#include <generator.hpp>
#include <global_conf.hpp>

/* external includes */
#include <string>
#include <stdexcept>
#include <iostream>

std::unordered_map<std::string, void (ArgParser::*)(ArgPack &)> ArgParser::flag_parsers_{
    {"-t", &ArgParser::ParseTestFlag_},
    {"-g", &ArgParser::ParseGeneratorFlag_},
    {"-d", &ArgParser::ParseWriteDot},
};


ArgParser::ArgParser(const int argc, char **argv): argc_(argc), argv_(argv) {
}

ArgPack ArgParser::ParseArgs() {
    ArgPack arg_pack{};

    if (argc_ < 1) {
        /* missing filename - its first positional argument */
        PrintHelp();
        throw std::runtime_error("Missing filename");
    }

    arg_pack.filename = argv_[0];

    for (cur_ = 1; cur_ < static_cast<size_t>(argc_);) {
        ProcessArgs_(arg_pack);
    }

    return arg_pack;
}

void ArgParser::PrintHelp() {
    std::cout << "Usage:\n"
            "<filename> [-g [generator_name]] [-t <test_name> ...]\n"
            "where:\n"
            "- \"filename\" is the name of the file to be processed or saved to\n"
            "- \"-g\" is an optional flag to generate data\n"
            "- \"-t\" is an optional flag to run a specific test\n\n";

    std::cout << "Available tests:\n";
    for (const auto test_names = Tester::GetTestNames(); const auto &test_name: test_names) {
        std::cout << "- \"" << test_name << "\"\n";
    }

    std::cout << "\nAvailable generators:\n";
    for (const auto generator_names = Generator::GetGeneratorNames(); const auto &generator_name: generator_names) {
        std::cout << "- \"" << generator_name << "\"\n";
    }

    std::cout.flush();
}

void ArgParser::ProcessArgs_(ArgPack &arg_pack) {
    const char *arg = argv_[cur_];

    if (!flag_parsers_.contains(arg)) {
        PrintHelp();
        throw std::runtime_error("Unknown flag: " + std::string(arg));
    }

    const auto &parser = flag_parsers_.at(arg);

    ++cur_;
    (this->*parser)(arg_pack);
}

void ArgParser::ParseTestFlag_(ArgPack &arg_pack) {
    arg_pack.run_tests = true;

    if (cur_ >= static_cast<size_t>(argc_)) {
        PrintHelp();
        throw std::runtime_error("Missing test name");
    }

    while (cur_ < static_cast<size_t>(argc_) && argv_[cur_][0] != '-') {
        arg_pack.test_names.emplace_back(argv_[cur_++]);
    }

    if (arg_pack.test_names.empty()) {
        throw std::runtime_error("Missing test name");
    }
}

void ArgParser::ParseGeneratorFlag_(ArgPack &arg_pack) {
    arg_pack.gen_data = true;

    if (cur_ >= static_cast<size_t>(argc_)) {
        PrintHelp();
        throw std::runtime_error("Missing generator name");
    }

    const char *generator_name = argv_[cur_++];
    arg_pack.generator_name = generator_name;
}

void ArgParser::ParseWriteDot([[maybe_unused]] ArgPack &arg_pack) {
    GlobalConfig.WriteDotFiles = true;
}
