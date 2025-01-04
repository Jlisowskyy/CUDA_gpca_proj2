#ifndef ARG_PARSER_HPP
#define ARG_PARSER_HPP

/* internal includes */

/* external includes */
#include <unordered_map>
#include <string>

struct ArgPack {
    const char *filename;
    bool gen_data;
    const char *generator_name;
    bool run_tests;
    const char *test_name;
};

class ArgParser {
    // ------------------------------
    // Class creation
    // ------------------------------

public:
    ArgParser(int argc, char **argv);

    ~ArgParser() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    [[nodiscard]] ArgPack ParseArgs();

    static void PrintHelp();

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    void ProcessArgs_(ArgPack &arg_pack);

    // ------------------------------
    // Flag parsers
    // ------------------------------

    void ParseTestFlag_(ArgPack &arg_pack);

    void ParseGeneratorFlag_(ArgPack &arg_pack);

    // ------------------------------
    // class fields
    // ------------------------------

    int argc_;
    char **argv_;

    size_t cur_{};

    static std::unordered_map<std::string, void (ArgParser::*)(ArgPack &)> flag_parsers_;
};

#endif //ARG_PARSER_HPP
