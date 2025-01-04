/* internal includes */
#include <arg_parser.hpp>
#include <dispatcher.hpp>

/* external includes */
#include <iostream>

static void main_(const int argc, char **argv) {
    ArgParser arg_parser(argc - 1, argv + 1);
    const auto arg_pack = arg_parser.ParseArgs();

    try {
        const Dispatcher dispatcher(arg_pack);
        dispatcher.Dispatch();
    } catch (const std::exception &e) {
        ArgParser::PrintHelp();
    }
}

int main(const int argc, char **argv) {
    try {
        main_(argc, argv);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Application failed with error: " << e.what() << std::endl;
        return 1;
    }
}
