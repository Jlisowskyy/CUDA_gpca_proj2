#ifndef RUNNER_HPP
#define RUNNER_HPP

/* internal includes */
#include <data.hpp>
#include <string>

class Runner {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    Runner() = default;

    ~Runner() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    void Run(const BinSequencePack &data, const std::string& filename);

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    // ------------------------------
    // class fields
    // ------------------------------
};

#endif //RUNNER_HPP
