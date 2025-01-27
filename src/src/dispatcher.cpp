/* internal includes */
#include <dispatcher.hpp>
#include <generator.hpp>
#include <runner.hpp>
#include <cassert>
#include <data_parser.hpp>
#include <hamming.cuh>
#include <tester.hpp>

Dispatcher::Dispatcher(const ArgPack &arg_pack) : arg_pack_(arg_pack) {
}

void Dispatcher::Dispatch() const {
    assert(arg_pack_.filename != nullptr);
    const DataParser data_parser{};

    if (!arg_pack_.gen_data && !arg_pack_.run_tests) {
        Runner runner{};
        runner.Run(data_parser.ParseData(arg_pack_.filename), arg_pack_.filename);
    }

    if (arg_pack_.gen_data) {
        /* generate data to allow run test in same command */
        assert(arg_pack_.generator_name != nullptr);

        Generator generator{};
        const auto generated_data = generator.GenerateData(arg_pack_.generator_name);
        data_parser.SaveData(generated_data, arg_pack_.filename);
    }

    if (arg_pack_.run_tests) {
        /* run tests possibly on previously generated data */
        assert(!arg_pack_.test_names.empty());

        Tester tester{};
        tester.RunTests(arg_pack_.test_names, data_parser.ParseData(arg_pack_.filename));
    }

}
